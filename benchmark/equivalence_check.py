"""Tier-1 equivalence runner for algorithm-preserving optimizations.

Runs the deploy NightSession deterministically (FakeClock + NightAdapter),
freezes + finalizes, and writes a content fingerprint (hashes + counters +
report metadata + JPEG hash).  Run once on baseline, once after an optimization,
then ``--compare before.json after.json``.

Any fingerprint difference means the optimization changed behavior -> reject.

Usage
-----
    python benchmark/equivalence_check.py --samples 1000 --out benchmark/out/equiv_before.json
    # ... apply optimization ...
    python benchmark/equivalence_check.py --samples 1000 --out benchmark/out/equiv_after.json
    python benchmark/equivalence_check.py --compare benchmark/out/equiv_before.json benchmark/out/equiv_after.json
"""
from __future__ import annotations

import argparse
import hashlib
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


def load_spec(cfg_path: Path, camera_id: str, args) -> NightLampSpec:
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam = next(c for c in d["cameras"] if c["id"] == camera_id)
    nl = cam["algorithms"]["night_lamp"]
    sampling = nl.get("sampling", {}) or {}
    memory = nl.get("memory", {}) or {}
    baseline = nl.get("baseline", {}) or {}
    period = nl.get("periodicity", {}) or {}
    align = nl.get("alignment", {}) or {}
    series_cap = memory.get("series_cap", None)
    if args.series_cap is not None:
        series_cap = None if args.series_cap == 0 else int(args.series_cap)
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=args.interval_ms or int(sampling.get("interval_ms", 160))),
        memory=MemorySpec(max_candidates=args.max_candidates or int(memory.get("max_candidates", 200)),
                          series_cap=series_cap),
        baseline=BaselineSpec(frames=args.baseline_frames or int(baseline.get("frames", 60)),
                              warmup_s=int(baseline.get("warmup_s", 600))),
        periodicity=PeriodicitySpec(
            period_step=int(period.get("period_step", 2)),
            lag_lo_s=float(period.get("lag_lo_s", 0.3)),
            lag_hi_s=float(period.get("lag_hi_s", 7.0)),
            peak_floor=float(period.get("peak_floor", 0.2)),
            min_peaks=int(period.get("min_peaks", 3)),
            gap_tol=int(period.get("gap_tol", 1)),
            onset_min=int(period.get("onset_min", 50))),
        alignment=AlignmentSpec(min_cc=float(align.get("min_cc", 0.5)),
                                on_low_cc=AlignmentStatus(align.get("on_low_cc", "degraded"))),
    )


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def fingerprint(sess, report) -> dict:
    d = sess._discovery
    ser = sess._series
    fp = {
        "n_seen": int(sess._n_seen),
        "series_n_points": int(ser.n_points),
        "series_pys": _h(ser._pys),
        "series_pxs": _h(ser._pxs),
        "series_starts": _h(ser._starts),
        "series_buf": _h(ser._buf[:ser.n_points, :ser._n_total]),
        "on": _h(sess._on),
        "peak": _h(sess._peak),
        "base": _h(sess._base),
        "n_components_seen": int(d.n_components_seen),
        "n_candidate_cores": int(d.n_candidate_cores),
        "n_dyn_cores": int(d.n_dyn_cores),
        "n_cand_dropped": int(d.n_cand_dropped),
        "n_dyn_dropped": int(d.n_dyn_dropped),
        "overflow": bool(d.overflow),
    }
    if report is not None:
        md = report.metadata or {}
        fp["report_status"] = report.status
        fp["report_meta"] = {k: md.get(k) for k in (
            "n_cand", "n_cand_total", "n_flash", "n_reflect",
            "candidate_overflow", "alignment", "night_state")}
        fp["jpeg_len"] = len(report.image_jpeg) if report.image_jpeg else 0
        fp["jpeg_sha"] = (_h(np.frombuffer(report.image_jpeg, np.uint8))
                          if report.image_jpeg else None)
    return fp


def run(args) -> dict:
    spec = load_spec(Path(args.config), args.camera, args)
    video = Path(args.video)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    sess = S.NightSession(args.camera, spec)
    clock = FakeClock(datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc))
    adapter = NightAdapter(sess, spec, clock)
    last = None
    t0 = time.perf_counter()
    while sess.sample_count < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        clock.advance(1.0 / fps)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())
        if time.perf_counter() - t0 > args.timebox_s:
            break
    cap.release()
    adapter.freeze()
    report = adapter.finalize(last, clock.wall())
    fp = fingerprint(sess, report)
    fp["_elapsed_s"] = round(time.perf_counter() - t0, 2)
    fp["_session_file"] = str(Path(S.__file__).resolve())
    return fp


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    keys = sorted(set(a) | set(b))
    diffs = []
    for k in keys:
        if k.startswith("_"):
            continue
        if a.get(k) != b.get(k):
            diffs.append((k, a.get(k), b.get(k)))
    print(f"before={a_path} ({a.get('_elapsed_s')}s)")
    print(f"after ={b_path} ({b.get('_elapsed_s')}s)")
    if not diffs:
        print("TIER-1 EQUIVALENT: all fingerprints match")
        return 0
    print(f"TIER-1 DIFFERS: {len(diffs)} field(s)")
    for k, av, bv in diffs:
        print(f"  {k}:\n    before={av}\n    after ={bv}")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="tmp/1002490_night_10min.mp4")
    ap.add_argument("--config", default="deploy/config.yaml")
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--timebox-s", type=float, default=900.0)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--series-cap", type=int, default=None)
    ap.add_argument("--baseline-frames", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    fp = run(a)
    print(json.dumps(fp, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(fp, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
