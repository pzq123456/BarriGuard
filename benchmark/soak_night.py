"""Full-night soak: prove overnight behavior is bounded on production config.

Deterministic (FakeClock) so the observation stream is exactly the production
cadence.  Frames are read from a night clip and looped to reach the target
number of observations (default 7 h at interval_ms=160 -> 157,500).

Instrumented per window: observation count, wall/obs (CPU proxy), RSS,
TemporalSeries bytes, discovery latency (max/mean), candidate counters,
overflow.  Ends with a freeze + finalize and its cost/status.

Limitation (explicit): with FakeClock the cadence is exact by construction, so
inter-observation jitter and queue depth are zero.  This soak proves CPU/RSS/
state boundedness over a long run; it does not reproduce real-time backlog
(that requires a live RTSP source).

Usage
-----
    python benchmark/soak_night.py --night-hours 0.2 --window-obs 1000
    python benchmark/soak_night.py --night-hours 7 --window-obs 2500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def load_spec(cfg_path: Path, camera_id: str, args) -> NightLampSpec:
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam = next(c for c in d["cameras"] if c["id"] == camera_id)
    nl = cam["algorithms"]["night_lamp"]
    s = nl.get("sampling", {}) or {}
    m = nl.get("memory", {}) or {}
    b = nl.get("baseline", {}) or {}
    p = nl.get("periodicity", {}) or {}
    a = nl.get("alignment", {}) or {}
    series_cap = m.get("series_cap", None)
    if args.series_cap is not None:
        series_cap = None if args.series_cap == 0 else int(args.series_cap)
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=args.interval_ms or int(s.get("interval_ms", 160))),
        memory=MemorySpec(max_candidates=args.max_candidates or int(m.get("max_candidates", 200)),
                          series_cap=series_cap),
        baseline=BaselineSpec(frames=args.baseline_frames or int(b.get("frames", 60)),
                              warmup_s=int(b.get("warmup_s", 600))),
        periodicity=PeriodicitySpec(
            period_step=int(p.get("period_step", 2)),
            lag_lo_s=float(p.get("lag_lo_s", 0.3)),
            lag_hi_s=float(p.get("lag_hi_s", 7.0)),
            peak_floor=float(p.get("peak_floor", 0.2)),
            min_peaks=int(p.get("min_peaks", 3)),
            gap_tol=int(p.get("gap_tol", 1)),
            onset_min=int(p.get("onset_min", 50))),
        alignment=AlignmentSpec(method=a.get("method", "ecc_translation"),
                                min_cc=float(a.get("min_cc", 0.5)),
                                on_low_cc=AlignmentStatus(a.get("on_low_cc", "degraded"))),
    )


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 1048576.0


def run(args) -> dict:
    spec = load_spec(Path(args.config), args.camera, args)
    interval_s = spec.sampling.interval_ms / 1000.0
    target = int(round(args.night_hours * 3600.0 / interval_s))
    if args.max_obs:
        target = min(target, args.max_obs)

    video = Path(args.video)
    sess = S.NightSession(args.camera, spec)
    clock = FakeClock(datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc))
    adapter = NightAdapter(sess, spec, clock)

    disc_times: list[float] = []
    disc_orig = sess._run_discovery

    def disc_wrap():
        t = time.perf_counter()
        try:
            return disc_orig()
        finally:
            disc_times.append((time.perf_counter() - t) * 1000.0)

    sess._run_discovery = disc_wrap

    windows = []
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    dt_frame = 1.0 / fps
    frames_read = 0
    loops = 0
    last = None
    win_start_obs = 0
    win_start_t = t0
    win_start_rss = rss0
    win_disc = 0
    stopped = False

    while sess.sample_count < target:
        ok, frame = cap.read()
        if not ok:
            loops += 1
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        frames_read += 1
        last = frame
        clock.advance(dt_frame)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())

        if time.perf_counter() - t0 > args.max_wall_s:
            print(f"  [max-wall] stop at samples={sess.sample_count}")
            stopped = True
            break
        if _rss_mb() > args.stop_rss_mb:
            print(f"  [rss-cap] stop at samples={sess.sample_count} rss={_rss_mb():.0f}MB")
            stopped = True
            break

        if sess.sample_count - win_start_obs >= args.window_obs:
            now = time.perf_counter()
            n_obs = sess.sample_count - win_start_obs
            windows.append(_window(sess, clock, n_obs, now - win_start_t,
                                   _rss_mb() - win_start_rss, win_disc, loops))
            win_start_obs = sess.sample_count
            win_start_t = now
            win_start_rss = _rss_mb()
            win_disc = len(disc_times)
            print(f"  obs={sess.sample_count:7d} wall={now - t0:7.1f}s "
                  f"rss={_rss_mb():7.1f}MB series={_series_mb(sess):7.1f}MB "
                  f"disc_max={max(disc_times[-50:], default=0):6.1f}ms "
                  f"cand={sess._discovery.n_candidate_cores} ovf={sess._discovery.overflow}")
    cap.release()

    try:
        adapter.freeze()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] freeze: {e}")

    t_fin = time.perf_counter()
    report = adapter.finalize(last, clock.wall())
    fin_s = time.perf_counter() - t_fin

    arr = np.array(disc_times, dtype=np.float64) if disc_times else np.zeros(1)
    result = {
        "mode": "soak",
        "video": str(video.resolve()),
        "night_hours": args.night_hours,
        "interval_ms": spec.sampling.interval_ms,
        "series_cap": spec.memory.series_cap,
        "max_candidates": spec.memory.max_candidates,
        "target_obs": target,
        "samples": int(sess.sample_count),
        "frames_read": frames_read,
        "video_loops": loops,
        "stopped_early": stopped,
        "wall_s": round(time.perf_counter() - t0, 1),
        "rss_start_mb": round(rss0, 1),
        "rss_end_mb": round(_rss_mb(), 1),
        "rss_growth_mb": round(_rss_mb() - rss0, 1),
        "series_end_mb": round(_series_mb(sess), 1),
        "discovery": {
            "calls": len(disc_times),
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
        },
        "finalize_s": round(fin_s, 1),
        "finalize_status": getattr(report, "status", None),
        "report_meta": {k: (report.metadata or {}).get(k) for k in (
            "n_cand", "n_cand_total", "n_flash", "n_reflect",
            "candidate_overflow", "sampling")},
        "discovery_counters": {
            "n_components_seen": sess._discovery.n_components_seen,
            "n_candidate_cores": sess._discovery.n_candidate_cores,
            "n_dyn_cores": sess._discovery.n_dyn_cores,
            "n_dropped_total": sess._discovery.n_dropped,
            "overflow": sess._discovery.overflow,
        },
        "windows": windows,
    }
    return result


def _series_mb(sess) -> float:
    return (sess._series.memory_bytes() / 1048576.0) if sess._series is not None else 0.0


def _window(sess, clock, n_obs, win_s, rss_delta, disc0, loops) -> dict:
    recent = sess._discovery
    return {
        "obs": int(sess.sample_count),
        "n_obs": int(n_obs),
        "win_s": round(win_s, 2),
        "ms_per_obs": round(win_s * 1000.0 / max(n_obs, 1), 3),
        "rss_mb": round(_rss_mb(), 1),
        "rss_delta_mb": round(rss_delta, 1),
        "series_mb": round(_series_mb(sess), 1),
        "obs_per_s": round(n_obs / max(win_s, 1e-9), 2),
        "candidate_cores": int(recent.n_candidate_cores),
        "dyn_cores": int(recent.n_dyn_cores),
        "components_seen": int(recent.n_components_seen),
        "dropped_total": int(recent.n_dropped),
        "overflow": bool(recent.overflow),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="tmp/1749_202609040100.mp4")
    ap.add_argument("--config", default="deploy/config.yaml")
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--night-hours", type=float, default=7.0)
    ap.add_argument("--window-obs", type=int, default=2500)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--series-cap", type=int, default=None)
    ap.add_argument("--baseline-frames", type=int, default=0)
    ap.add_argument("--max-obs", type=int, default=0)
    ap.add_argument("--max-wall-s", type=float, default=5400.0)
    ap.add_argument("--stop-rss-mb", type=float, default=8192.0)
    ap.add_argument("--out", default="benchmark/out")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    r = run(a)
    fp = outdir / f"soak_{a.camera}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    fp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: r[k] for k in r if k != "windows"}, indent=2, ensure_ascii=False))
    print(f"saved {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
