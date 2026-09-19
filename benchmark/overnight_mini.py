"""Deterministic mini-night: exercise the hourly-burst runtime end to end.

Drives the production ``NightSession`` + overnight ``NightAdapter`` with a
FakeClock and a local mp4, at an accelerated cadence.  Proves, without waiting
7 h:

  * a heatmap is produced at every burst end (report_type=night_heatmap_burst),
  * the spatial accumulator keeps accumulating across bursts,
  * the temporal state is reset at each burst start,
  * observation drops are counted (no silent drops),
  * the morning finalize produces the final night_heatmap and releases.

Usage
-----
    python benchmark/overnight_mini.py --bursts 3 --cadence-min 1 --burst-s 20
    python benchmark/overnight_mini.py --out tmp/overnight_mini_out
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from night_lamp.adapter import NightAdapter  # noqa: E402
from night_lamp.session import NightSession  # noqa: E402
from server.contracts import (  # noqa: E402
    AlignmentSpec, BaselineSpec, MemorySpec, NightLampSpec, OvernightSpec,
    PeriodicitySpec, SamplingSpec,
)
from server.output import ReportStore  # noqa: E402
from server.schedule import FakeClock  # noqa: E402

TZ = timezone(timedelta(hours=8))


def build_spec(cadence_min: int, burst_s: int) -> NightLampSpec:
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=160),
        memory=MemorySpec(max_candidates=200, series_cap=None),
        baseline=BaselineSpec(frames=60, warmup_s=600),
        periodicity=PeriodicitySpec(period_step=2, lag_lo_s=0.3, lag_hi_s=7.0,
                                    peak_floor=0.2, min_peaks=3, gap_tol=1,
                                    onset_min=50),
        alignment=AlignmentSpec(method="ecc_translation", min_cc=0.5),
        overnight=OvernightSpec(enabled=True, cadence_minutes=cadence_min,
                                burst_seconds=burst_s),
    )


def run(video: Path, out_dir: Path, cadence_min: int, burst_s: int,
        bursts: int) -> dict:
    spec = build_spec(cadence_min, burst_s)
    interval_s = spec.sampling.interval_ms / 1000.0
    step = interval_s * 1.05          # slightly above interval: no interval drops

    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    total_s = cadence_min * 60 * bursts

    clock = FakeClock(datetime(2026, 9, 18, 20, 0, tzinfo=TZ))
    session = NightSession("1749", spec)
    adapter = NightAdapter(session, spec, clock)
    store = ReportStore(str(out_dir))
    store.root.mkdir(parents=True, exist_ok=True)

    series_ids, series_refs = [], []
    orig_begin = session.begin_burst

    def begin(ts):
        series_refs.append(session._series)   # hold a ref so id() cannot be reused
        series_ids.append(id(session._series))
        return orig_begin(ts)

    session.begin_burst = begin  # harness instrumentation, no production change

    reports, spatial_trace = [], []
    t = 0.0
    frames = 0
    while t < total_s:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        frames += 1
        clock.advance(step)
        t += step
        adapter.on_frame(frame, clock.wall(), clock.monotonic())
        for rep in adapter.take_reports():
            reports.append(rep)
            spatial_trace.append({
                "burst": len(reports),
                "created_at": rep.created_at,
                "report_type": rep.report_type,
                "burst_samples": rep.metadata["burst"]["samples"],
                "cumulative_observations": rep.metadata["sampling"]["actual_samples"],
                "n_cand": rep.metadata.get("n_cand"),
                "n_flash": rep.metadata.get("n_flash"),
                "status": rep.status,
                "path": store.save(rep),
            })
            print(f"[burst] #{len(reports)} end={rep.created_at} "
                  f"samples={rep.metadata['burst']['samples']} "
                  f"cum_obs={rep.metadata['sampling']['actual_samples']} "
                  f"cand={rep.metadata.get('n_cand')} "
                  f"flash={rep.metadata.get('n_flash')} status={rep.status}")
    cap.release()

    adapter.freeze()
    final = adapter.finalize(None, clock.wall())
    final_path = store.save(final)
    adapter.release()

    return {
        "frames_read": frames,
        "burst_reports": len(reports),
        "series_ids": series_ids,
        "series_reset": (len(series_refs) == bursts
                         and len(set(series_ids)) == bursts),
        "spatial_trace": spatial_trace,
        "final": {
            "created_at": final.created_at,
            "report_type": final.report_type,
            "status": final.status,
            "path": final_path,
            "cumulative_observations": final.metadata["sampling"]["actual_samples"],
        },
        "adapter_stats": adapter.stats(),
        "memory_mb": session._memory_estimate_mb(),
        "released": session._released,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic overnight mini-night")
    ap.add_argument("--video", default="tmp/1002490_night_10min.mp4")
    ap.add_argument("--out", default="tmp/overnight_mini_out")
    ap.add_argument("--cadence-min", type=int, default=1)
    ap.add_argument("--burst-s", type=int, default=20)
    ap.add_argument("--bursts", type=int, default=3)
    a = ap.parse_args(argv)

    video = Path(a.video)
    if not video.is_file():
        raise FileNotFoundError(video)
    out = Path(a.out)
    print(f"[mini] video={video} cadence={a.cadence_min}min burst={a.burst_s}s "
          f"bursts={a.bursts}")
    r = run(video, out, a.cadence_min, a.burst_s, a.bursts)

    print("\n=== summary ===")
    for row in r["spatial_trace"]:
        print(f"  burst #{row['burst']}: cum_obs={row['cumulative_observations']} "
              f"cand={row['n_cand']} flash={row['n_flash']} -> {row['path']}")
    print(f"  temporal resets: {r['series_reset']} (ids={len(set(r['series_ids']))})")
    print(f"  final: {r['final']['report_type']} status={r['final']['status']} "
          f"obs={r['final']['cumulative_observations']} -> {r['final']['path']}")
    print(f"  adapter: {r['adapter_stats']}")
    print(f"  session memory={r['memory_mb']}MB released={r['released']}")

    assert r["burst_reports"] == a.bursts, r["burst_reports"]
    assert r["series_reset"], r["series_ids"]
    assert r["final"]["report_type"] == "night_heatmap"
    assert r["adapter_stats"]["dropped_burst"] > 0
    assert r["released"]
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
