"""C3 pre-study: how much temporal history does the EXISTING periodicity need?

Feeds a deterministic night segment, then replays the session's own `flashing`
decision (onsets gate + ac_limited + clean_train) on truncated tails of every
tracked series, comparing to the full-length decision.

This does NOT change production. It answers guardrail #2:
the ring size must come from measured equivalence, not a guessed 128/256.

Usage
-----
    python benchmark/periodicity_window_probe.py --samples 3000
"""
from __future__ import annotations

import argparse
import json
import sys
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
from night_lamp.periodicity import ac_limited, clean_train, onsets_of  # noqa: E402
from night_lamp.tools import nightly_map as nm  # noqa: E402

WINDOWS = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]


def load_spec(cfg_path: Path, camera_id: str, args) -> NightLampSpec:
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam = next(c for c in d["cameras"] if c["id"] == camera_id)
    nl = cam["algorithms"]["night_lamp"]
    p = nl.get("periodicity", {}) or {}
    s = nl.get("sampling", {}) or {}
    m = nl.get("memory", {}) or {}
    b = nl.get("baseline", {}) or {}
    a = nl.get("alignment", {}) or {}
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=args.interval_ms or int(s.get("interval_ms", 160))),
        memory=MemorySpec(max_candidates=args.max_candidates or int(m.get("max_candidates", 200)),
                          series_cap=None),
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
        alignment=AlignmentSpec(min_cc=float(a.get("min_cc", 0.5)),
                                on_low_cc=AlignmentStatus(a.get("on_low_cc", "degraded"))),
    )


def flashing(series, base_val, lag_lo, lag_hi, period) -> bool:
    on = (series.astype(np.float64) - base_val) >= nm.ON_DELTA
    if onsets_of(on) < period.onset_min:
        return False
    curve = ac_limited(on, lag_lo, lag_hi)[2]
    return clean_train(curve, lag_lo, period.peak_floor,
                       period.min_peaks, period.gap_tol)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="tmp/1002490_night_10min.mp4")
    ap.add_argument("--config", default="deploy/config.yaml")
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--timebox-s", type=float, default=900.0)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--baseline-frames", type=int, default=0)
    ap.add_argument("--out", default="benchmark/out/periodicity_window_probe.json")
    a = ap.parse_args(argv)

    spec = load_spec(Path(a.config), a.camera, a)
    cap = cv.VideoCapture(a.video)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    sess = S.NightSession(a.camera, spec)
    clock = FakeClock(datetime(2026, 9, 18, 22, 0, tzinfo=timezone.utc))
    adapter = NightAdapter(sess, spec, clock)
    import time
    t0 = time.perf_counter()
    while sess.sample_count < a.samples:
        ok, frame = cap.read()
        if not ok:
            break
        clock.advance(1.0 / fps)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())
        if time.perf_counter() - t0 > a.timebox_s:
            break
    cap.release()
    sess.freeze()

    rate = sess._design_rate_hz() or 1.0
    lag_lo = max(1, int(round(spec.periodicity.lag_lo_s * rate)))
    lag_hi = max(lag_lo + 1, int(round(spec.periodicity.lag_hi_s * rate)))

    ser = sess._series
    rows = []
    for r in range(ser.n_points):
        s = ser.series(r)
        if s.size < 4:
            continue
        y, x = int(ser._pys[r]), int(ser._pxs[r])
        rows.append((s, float(sess._base[y, x])))

    full = np.array([flashing(s, b, lag_lo, lag_hi, spec.periodicity)
                     for s, b in rows], bool)
    n_full_flash = int(full.sum())

    report = {"samples": sess.sample_count, "rate_hz": round(rate, 4),
              "lag_lo": lag_lo, "lag_hi": lag_hi,
              "onset_min": spec.periodicity.onset_min,
              "tracked_rows": len(rows), "full_flashers": n_full_flash,
              "max_len": int(max((s.size for s, _ in rows), default=0)),
              "windows": {}}

    print(f"samples={sess.sample_count} rate={rate:.3f}Hz lag=[{lag_lo},{lag_hi}] "
          f"tracked={len(rows)} full_flashers={n_full_flash} "
          f"max_len={report['max_len']}")

    for W in WINDOWS:
        tested = [(s, b, full[i]) for i, (s, b) in enumerate(rows) if s.size >= W]
        if not tested:
            continue
        agree = fn = fp = onsets_gate_diff = 0
        b_agree = b_fn = b_fp = 0
        for s, b, ffull in tested:
            tail = s[-W:]
            on_tail = (tail.astype(np.float64) - b) >= nm.ON_DELTA
            on_full = (s.astype(np.float64) - b) >= nm.ON_DELTA
            # mode A: naive truncation of the whole flashing rule
            dec = flashing(tail, b, lag_lo, lag_hi, spec.periodicity)
            if dec == ffull:
                agree += 1
            elif ffull and not dec:
                fn += 1
            else:
                fp += 1
            # mode B: keep the onset gate as an online full-history count,
            # truncate only the autocorrelation window.
            if onsets_of(on_full) < spec.periodicity.onset_min:
                dec_b = False
            else:
                curve = ac_limited(on_tail, lag_lo, lag_hi)[2]
                dec_b = clean_train(curve, lag_lo, spec.periodicity.peak_floor,
                                    spec.periodicity.min_peaks,
                                    spec.periodicity.gap_tol)[0]
            if dec_b == ffull:
                b_agree += 1
            elif ffull and not dec_b:
                b_fn += 1
            else:
                b_fp += 1
            if (onsets_of(on_tail) < spec.periodicity.onset_min) != \
               (onsets_of(on_full) < spec.periodicity.onset_min):
                onsets_gate_diff += 1
        n = len(tested)
        report["windows"][str(W)] = {
            "tested": n,
            "A_agreement": round(agree / n, 4), "A_false_neg": fn, "A_false_pos": fp,
            "B_agreement": round(b_agree / n, 4), "B_false_neg": b_fn, "B_false_pos": b_fp,
            "onset_gate_flips": onsets_gate_diff,
        }
        print(f"  W={W:5d} A(naive) agree={agree/n:5.2f} fn={fn:3d} fp={fp:2d} | "
              f"B(online-onset) agree={b_agree/n:5.2f} fn={b_fn:3d} fp={b_fp:2d} | "
              f"gate_flips={onsets_gate_diff}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
