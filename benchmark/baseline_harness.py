"""Performance baseline harness for the deploy night_lamp pipeline.

NOTHING in production is modified.  This script only *wraps* production objects
at runtime to attribute wall time, and drives them through the production
``NightAdapter`` + a ``FakeClock`` so the sampling decisions are deterministic
(no ``time.monotonic()`` dependence).

Canonical baseline (locked for this exercise)
---------------------------------------------
* Algorithm: ``deploy/app/night_lamp/session.py`` (``NightSession``) plus
  ``tools/nightly_map.py`` / ``periodicity.py``.
* Config:    ``deploy/config.yaml`` -> cameras[1749].algorithms.night_lamp.
* Golden reference: the deploy session's own ``Report`` (not the research
  pipeline, not ``night_lamp/detector.py``).

Experiments
-----------
A. ``--mode pipeline`` : stage-level wall time of the per-sample path
   (accumulate / warmup / process / prep / series / discovery) + micro-ops.
B. ``--mode components``: component-extraction micro-benchmark, quantifying the
   ``np.nonzero(lab == c)`` loop vs a single-pass grouped extraction.
C. decoder/ffmpeg cost is measured separately by ``benchmark_ffmpeg.py``.

Usage
-----
    python benchmark/baseline_harness.py --samples 1500
    python benchmark/baseline_harness.py --mode components
    python benchmark/baseline_harness.py --video tmp/1002490_night_10min.mp4 \
        --samples 400 --timebox-s 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from server.contracts import (  # noqa: E402
    AlignmentSpec, AlignmentStatus, BaselineSpec, MemorySpec, NightLampSpec,
    PeriodicitySpec, SamplingSpec,
)
from server.schedule import FakeClock  # noqa: E402
from night_lamp.adapter import NightAdapter  # noqa: E402
from night_lamp import session as S  # noqa: E402
from night_lamp.tools import nightly_map as nm  # noqa: E402

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

DEFAULT_VIDEO = "tmp/1002490_night_10min.mp4"
DEFAULT_CONFIG = "deploy/config.yaml"
CAMERA = "1749"


# --------------------------------------------------------------------------- spec
def load_spec(cfg_path: Path, camera_id: str, args) -> NightLampSpec:
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam = next(c for c in d["cameras"] if c["id"] == camera_id)
    nl = cam["algorithms"]["night_lamp"]

    sampling = nl.get("sampling", {}) or {}
    memory = nl.get("memory", {}) or {}
    baseline = nl.get("baseline", {}) or {}
    period = nl.get("periodicity", {}) or {}
    align = nl.get("alignment", {}) or {}

    interval = args.interval_ms or int(sampling.get("interval_ms", 160))
    max_cand = args.max_candidates or int(memory.get("max_candidates", 200))
    if args.series_cap is not None:
        series_cap = None if args.series_cap == 0 else int(args.series_cap)
    else:
        series_cap = memory.get("series_cap", None)
    base_frames = args.baseline_frames or int(baseline.get("frames", 60))

    on_low = align.get("on_low_cc", "degraded")
    return NightLampSpec(
        calibration=nl.get("calibration", ""),
        night_gate=nl.get("night_gate", {}) or {},
        sampling=SamplingSpec(interval_ms=interval),
        memory=MemorySpec(max_candidates=max_cand, series_cap=series_cap),
        baseline=BaselineSpec(frames=base_frames,
                              warmup_s=int(baseline.get("warmup_s", 600))),
        periodicity=PeriodicitySpec(
            period_step=int(period.get("period_step", 2)),
            lag_lo_s=float(period.get("lag_lo_s", 0.3)),
            lag_hi_s=float(period.get("lag_hi_s", 7.0)),
            peak_floor=float(period.get("peak_floor", 0.2)),
            min_peaks=int(period.get("min_peaks", 3)),
            gap_tol=int(period.get("gap_tol", 1)),
            onset_min=int(period.get("onset_min", 50))),
        alignment=AlignmentSpec(method=align.get("method", "ecc_translation"),
                                min_cc=float(align.get("min_cc", 0.5)),
                                on_low_cc=AlignmentStatus(on_low)),
    )


# --------------------------------------------------------------------------- timing
class Timer:
    def __init__(self):
        self.total: dict[str, float] = {}
        self.count: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        self.total[name] = self.total.get(name, 0.0) + dt
        self.count[name] = self.count.get(name, 0) + 1


def patch(obj, name: str, timer: Timer, key: str):
    """Wrap ``obj.name`` to accumulate wall time under ``key``. Returns restore."""
    orig = getattr(obj, name)

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            timer.add(key, time.perf_counter() - t)

    setattr(obj, name, wrapper)
    return lambda: setattr(obj, name, orig)


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1048576.0


# --------------------------------------------------------------------------- experiment A
def run_pipeline(args) -> dict:
    t_start = time.perf_counter()
    print(f"[PERF] START wall_clock={datetime.now().isoformat()}")
    session_file = Path(S.__file__).resolve()
    assert "deploy" in str(session_file), f"wrong night_lamp package: {session_file}"
    print(f"[canonical] night_lamp.session -> {session_file}")

    spec = load_spec(Path(args.config), args.camera, args)
    video = Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")

    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    dt_frame = 1.0 / fps

    sess = S.NightSession(args.camera, spec)
    clock = FakeClock(datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc))
    adapter = NightAdapter(sess, spec, clock)

    timer = Timer()
    passes: list[dict] = []
    restores = []

    # top-level phases (instance-level => no global side effects)
    restores.append(patch(sess, "accumulate", timer, "accumulate.total"))
    restores.append(patch(sess, "_finish_warmup", timer, "accumulate.warmup"))
    restores.append(patch(sess, "_process", timer, "accumulate.process"))
    restores.append(patch(sess._series, "append", timer, "accumulate.series"))
    restores.append(patch(sess._series, "aligned_matrix", timer, "finalize.aligned_matrix"))

    # discovery pass instrumentation (records per-pass counters)
    disc_orig = sess._run_discovery
    phase = {"name": "accumulate.discovery"}

    def disc_wrap():
        t = time.perf_counter()
        try:
            return disc_orig()
        finally:
            dt = time.perf_counter() - t
            timer.add(phase["name"], dt)
            d = sess._discovery
            passes.append({
                "sample_index": int(sess._n_seen),
                "duration_ms": round(dt * 1000, 2),
                "components_seen_total": int(d.n_components_seen),
                "candidate_cores": int(d.n_candidate_cores),
                "dyn_cores": int(d.n_dyn_cores),
                "dropped_total": int(d.n_dropped),
                "overflow": bool(d.overflow),
            })

    sess._run_discovery = disc_wrap
    restores.append(lambda: setattr(sess, "_run_discovery", disc_orig))

    # module-level wraps (shared across phases; labelled explicitly)
    mod_wraps = [
        (nm, "prep", "call.prep"),
        (nm, "split_comps", "call.split_comps"),
        (nm, "dyn_samples", "call.dyn_samples"),
        # finalize sub-steps
        (nm, "dim_map", "finalize.dim_map"),
        (nm, "despeckle", "finalize.despeckle"),
        (nm, "heat_bgr", "finalize.heat_bgr"),
        (nm, "align_day", "finalize.align_day"),
        (nm, "compose", "finalize.compose"),
        (nm, "night_bg", "finalize.night_bg"),
        (nm, "draw_rings", "finalize.draw_rings"),
        (S, "ac_limited", "finalize.periodicity"),
        (S, "onsets_of", "finalize.periodicity"),
        (S, "clean_train", "finalize.periodicity"),
    ]
    for mod, name, key in mod_wraps:
        restores.append(patch(mod, name, timer, key))
    restores.append(patch(S.cv, "imencode", timer, "finalize.imencode"))

    restores.append(patch(sess, "finalize", timer, "finalize.total"))

    max_samples = int(args.samples)
    deadline = time.perf_counter() + float(args.timebox_s)
    n_frames = 0
    t_read = 0.0
    rss_start = _rss_mb()
    rss_peak = rss_start
    last_frame = None
    t_loop0 = time.perf_counter()
    warm_at = None
    timeout = False

    while sess.sample_count < max_samples:
        t = time.perf_counter()
        ok, frame = cap.read()
        t_read += time.perf_counter() - t
        if not ok:
            break
        last_frame = frame
        clock.advance(dt_frame)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())
        n_frames += 1
        if warm_at is None and sess._warm_done:
            warm_at = sess.sample_count
        if n_frames % 100 == 0:
            rss = _rss_mb()
            rss_peak = max(rss_peak, rss)
            print(f"  frames={n_frames} samples={sess.sample_count} "
                  f"elapsed={time.perf_counter() - t_loop0:6.1f}s "
                  f"rss={rss:6.1f}MB discovery_calls={timer.count.get('accumulate.discovery', 0)}")
        if time.perf_counter() > deadline:
            timeout = True
            print(f"  [timebox] stop at samples={sess.sample_count}")
            break

    t_loop = time.perf_counter() - t_loop0
    samples = sess.sample_count
    cap.release()

    phase["name"] = "freeze.discovery"
    try:
        adapter.freeze()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] freeze failed: {e}")

    # finalize (use the last night frame as the base so align_day is exercised;
    # alignment *quality* is meaningless without a real day image -> recorded).
    base = last_frame if (last_frame is not None and not args.no_ecc) else None
    t = time.perf_counter()
    try:
        report = adapter.finalize(base, clock.wall())
        fin_ok = True
    except Exception as e:  # noqa: BLE001
        report = None
        fin_ok = False
        print(f"  [warn] finalize failed: {type(e).__name__}: {e}")
    t_finalize_wall = time.perf_counter() - t

    md = (report.metadata if report is not None else {}) or {}
    rss_end = _rss_mb()
    mem_est = sess._memory_estimate_mb() if hasattr(sess, "_memory_estimate_mb") else None
    series_bytes = sess._series.memory_bytes() if sess._series is not None else None

    image_path = None
    if args.save_image and report is not None and report.image_jpeg:
        Path(args.save_image).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_image).write_bytes(report.image_jpeg)
        image_path = str(Path(args.save_image).resolve())

    total_wall = time.perf_counter() - t_start
    eff_hz = sess._rate_hz() if hasattr(sess, "_rate_hz") else 0.0
    print(f"[PERF] INPUT={video.resolve()}")
    print(f"[PERF] FRAMES={n_frames}")
    print(f"[PERF] OBSERVATIONS={samples}")
    print(f"[PERF] EFFECTIVE_HZ={eff_hz:.4f}")
    print(f"[PERF] FINALIZE={t_finalize_wall:.3f} (heatmap is produced/encoded here)")
    print(f"[PERF] HEATMAP_GENERATION={t_finalize_wall:.3f}")
    print(f"[PERF] HEATMAP_PATH={image_path}")
    print(f"[PERF] END wall_clock={datetime.now().isoformat()}")
    print(f"[PERF] TOTAL_WALL={total_wall:.3f}")

    for r in restores:
        try:
            r()
        except Exception:  # noqa: BLE001
            pass

    result = {
        "mode": "pipeline",
        "canonical_session": str(session_file),
        "config": str(Path(args.config).resolve()),
        "video": str(video.resolve()),
        "fps": round(fps, 4),
        "interval_ms": spec.sampling.interval_ms,
        "max_candidates": spec.memory.max_candidates,
        "series_cap": spec.memory.series_cap,
        "baseline_frames": spec.baseline.frames,
        "frames_read": n_frames,
        "samples": samples,
        "warmup_sample": warm_at,
        "timebox_hit": timeout,
        "read_s": round(t_read, 3),
        "loop_wall_s": round(t_loop, 3),
        "finalize_wall_s": round(t_finalize_wall, 3),
        "finalize_ok": fin_ok,
        "report_status": (report.status if report is not None else None),
        "report_has_jpeg": bool(report is not None and report.image_jpeg),
        "rss_start_mb": round(rss_start, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "rss_end_mb": round(rss_end, 1),
        "session_memory_estimate_mb": mem_est,
        "series_bytes": series_bytes,
        "timers": {k: {"ms": round(v * 1000, 2), "calls": timer.count.get(k, 0)}
                   for k, v in timer.total.items()},
        "discovery_passes": passes,
    }
    _add_derived(result)
    if report is not None:
        result["report_meta_subset"] = {k: md.get(k) for k in (
            "n_cand", "n_cand_total", "n_flash", "n_reflect", "candidate_overflow",
            "memory_mb", "sampling", "online")}
    return result


def _add_derived(result: dict) -> None:
    tm = result["timers"]
    samples = max(result["samples"], 1)

    def ms(key):
        return tm.get(key, {}).get("ms", 0.0)

    acc = ms("accumulate.total")
    result["derived"] = {
        "ms_per_sample_accumulate": round(acc / samples, 3),
        "ms_per_sample_process": round(ms("accumulate.process") / samples, 3),
        "ms_per_sample_prep": round(ms("call.prep") / samples, 3),
        "ms_per_sample_series": round(ms("accumulate.series") / samples, 3),
        "accumulate_other_ms": round(acc - ms("accumulate.process")
                                     - ms("accumulate.discovery"), 2),
        "discovery_freeze_ms": round(ms("freeze.discovery"), 2),
        "process_rest_ms": round(ms("accumulate.process") - ms("call.prep")
                                 - ms("accumulate.series"), 2),
    }


# --------------------------------------------------------------------------- experiment B
def bench_components(args) -> dict:
    """Quantify O(C * H*W) label scan vs a single-pass grouped extraction."""
    spec = load_spec(Path(args.config), args.camera, args)
    video = Path(args.video)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)

    sess = S.NightSession(args.camera, spec)
    clock = FakeClock(datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc))
    adapter = NightAdapter(sess, spec, clock)
    for i in range(max(args.samples, 400)):
        ok, frame = cap.read()
        if not ok:
            break
        clock.advance(1.0 / fps)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())
    cap.release()
    n = max(sess._n_seen, 1)
    duty = (sess._on / float(n)).astype(np.float32)

    dyn, lab, stats, cand = nm.split_comps(duty)
    cand = sorted(cand, key=lambda c: -stats[c, cv.CC_STAT_AREA])
    cap_c = args.max_candidates or spec.memory.max_candidates
    used = cand[:cap_c]

    H, W = duty.shape

    # current implementation: one full-frame scan per component
    t = time.perf_counter()
    cur_pixels = 0
    for c in cand:  # note: production discovery loops over ALL candidates
        ys, xs = np.nonzero(lab == c)
        cur_pixels += ys.size
    t_current_all = time.perf_counter() - t

    t = time.perf_counter()
    for c in used:
        ys, xs = np.nonzero(lab == c)
    t_current_used = time.perf_counter() - t

    # candidate A: single nonzero + stable argsort grouping
    t = time.perf_counter()
    ys_all, xs_all = np.nonzero(lab)
    labels = lab[ys_all, xs_all]
    order = np.argsort(labels, kind="stable")
    ys_s, xs_s = ys_all[order], xs_all[order]
    cnt = np.bincount(labels, minlength=int(lab.max()) + 1)
    offs = np.concatenate([[0], np.cumsum(cnt)])
    slices = {int(c): (ys_s[offs[c]:offs[c + 1]], xs_s[offs[c]:offs[c + 1]])
              for c in used}
    t_grouped = time.perf_counter() - t

    # candidate B: per-component bounding-box only (no full pixel list)
    t = time.perf_counter()
    bboxes = {int(c): (int(stats[c, cv.CC_STAT_LEFT]), int(stats[c, cv.CC_STAT_TOP]),
                       int(stats[c, cv.CC_STAT_WIDTH]), int(stats[c, cv.CC_STAT_HEIGHT]))
              for c in used}
    t_bbox = time.perf_counter() - t

    return {
        "mode": "components",
        "video": str(video.resolve()),
        "samples": n,
        "shape": [int(H), int(W)],
        "n_components_total": int(len(cand)),
        "n_components_used": int(len(used)),
        "n_dyn_pixels": int(dyn.sum()),
        "current_all_candidates_ms": round(t_current_all * 1000, 2),
        "current_used_candidates_ms": round(t_current_used * 1000, 2),
        "grouped_extract_ms": round(t_grouped * 1000, 3),
        "bbox_only_ms": round(t_bbox * 1000, 4),
        "ms_per_component_current": round(t_current_all * 1000 / max(len(cand), 1), 3),
        "speedup_grouped_vs_current": round(t_current_all / max(t_grouped, 1e-9), 2),
        "speedup_bbox_vs_current": round(t_current_all / max(t_bbox, 1e-9), 2),
    }


# --------------------------------------------------------------------------- micro ops
def bench_micro(args) -> dict:
    video = Path(args.video)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame")
    h, w = frame.shape[:2]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = nm.osd_mask(h, w)
    base = gray.astype(np.float32)
    on = np.zeros((h, w), np.float32)
    peak = np.zeros((h, w), np.float32)
    reps = int(args.reps)

    def timeit(fn):
        fn()  # warm
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps * 1000

    def do_cvt():
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def do_median():
        return np.median(gray)

    def do_prep():
        return nm.prep(gray, mask, 3)

    g16 = gray.astype(np.int16)

    def do_sub_thr_max_full():
        excess = g16 - base
        np.add(on, excess >= nm.ON_DELTA, out=on)
        np.maximum(peak, excess, out=peak)

    def do_sub_thr_max_nocopy():
        excess = np.subtract(g16, base)
        np.add(on, excess >= nm.ON_DELTA, out=on)
        np.maximum(peak, excess, out=peak)

    return {
        "mode": "micro",
        "shape": [int(h), int(w)],
        "reps": reps,
        "cvtColor_ms": round(timeit(do_cvt), 3),
        "median_gray_ms": round(timeit(do_median), 3),
        "prep_ms": round(timeit(do_prep), 3),
        "sub_threshold_max_ms": round(timeit(do_sub_thr_max_full), 3),
    }


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="night_lamp performance baseline harness")
    ap.add_argument("--mode", choices=("pipeline", "components", "micro", "all"),
                    default="pipeline")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--camera", default=CAMERA)
    ap.add_argument("--samples", type=int, default=1500, help="emitted samples to collect")
    ap.add_argument("--timebox-s", type=float, default=600.0)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--series-cap", type=int, default=None,
                    help="0 => None (full sequence, production default)")
    ap.add_argument("--baseline-frames", type=int, default=0)
    ap.add_argument("--no-ecc", action="store_true",
                    help="skip align_day (base frame = None)")
    ap.add_argument("--save-image", default=None,
                    help="write the finalize Report JPEG here (the heatmap)")
    ap.add_argument("--out", default="benchmark/out")
    a = ap.parse_args(argv)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    if a.mode in ("pipeline", "all"):
        results["pipeline"] = run_pipeline(a)
    if a.mode in ("micro", "all"):
        results["micro"] = bench_micro(a)
    if a.mode in ("components", "all"):
        results["components"] = bench_components(a)

    tag = time.strftime("%Y%m%d_%H%M%S")
    fp = outdir / f"baseline_{a.camera}_{a.mode}_{tag}.json"
    fp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {fp}")
    _print_summary(results)
    return 0


def _print_summary(results: dict) -> None:
    for name, r in results.items():
        print(f"\n=== {name} ===")
        if r.get("mode") == "pipeline":
            print(f"  frames={r['frames_read']} samples={r['samples']} "
                  f"fps={r['fps']} warp={r['loop_wall_s']}s finalize={r['finalize_wall_s']}s")
            print(f"  rss start/peak/end = {r['rss_start_mb']}/{r['rss_peak_mb']}/"
                  f"{r['rss_end_mb']} MB")
            for k, v in sorted(r["timers"].items(), key=lambda x: -x[1]["ms"]):
                print(f"    {k:28s} {v['ms']:10.2f} ms  x{v['calls']}")
            print("  derived:", json.dumps(r.get("derived", {}), ensure_ascii=False))
        elif r.get("mode") == "components":
            print(json.dumps(r, indent=2, ensure_ascii=False))
        elif r.get("mode") == "micro":
            print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
