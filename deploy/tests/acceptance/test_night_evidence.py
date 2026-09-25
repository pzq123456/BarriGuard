"""Night flash evidence lifecycle (P0) + sampling decoupling (T4).

Freezes the Phase-0 semantics: the morning map is the OR of per-hour flash
presence, persisted as derived evidence; continuous mode must not fabricate it.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "app"
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402

TZ = timezone(timedelta(hours=8))
_CACHE = {}


def _spec(overnight=True):
    from server.contracts import (AlignmentSpec, BaselineSpec, MemorySpec,
                                  NightLampSpec, OvernightSpec, PeriodicitySpec,
                                  SamplingSpec)
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=160),
        memory=MemorySpec(max_candidates=200, series_cap=None),
        baseline=BaselineSpec(frames=60, warmup_s=600),
        periodicity=PeriodicitySpec(period_step=2, lag_lo_s=0.3, lag_hi_s=7.0,
                                    peak_floor=0.2, min_peaks=3, gap_tol=1,
                                    onset_min=50),
        alignment=AlignmentSpec(min_cc=0.5),
        overnight=OvernightSpec(enabled=overnight, cadence_minutes=60,
                                burst_seconds=120, anchor="wall"),
    )


def _feed_session(sess, n):
    video = H.ensure_video(None)
    cap = cv.VideoCapture(str(video))
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        sess.accumulate(f, i * 0.16)
    cap.release()


def _run_wall(ctx, buckets=3):
    key = ("wall", ctx.video, buckets)
    if key in _CACHE:
        return _CACHE[key]
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server.schedule import FakeClock

    spec = _spec(True)
    cap = cv.VideoCapture(str(H.ensure_video(ctx.video)))
    dt = 1.0 / float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    clock = FakeClock(datetime(2026, 9, 21, 20, 0, tzinfo=TZ))
    ad = NightAdapter(NightSession("1749", spec), spec, clock)
    windows = [(k * 3600.0, k * 3600.0 + 120.0) for k in range(buckets + 1)]
    horizon = buckets * 3600.0 + 1.0
    reports, seq, t = [], 0, clock.monotonic()
    while t < horizon:
        if any(a <= t < b for a, b in windows):
            ok, frame = cap.read()
            if not ok:
                break
            seq += 1
            clock.advance(dt)
            ad.on_frame(frame, clock.wall(), clock.monotonic(), frame_id=seq)
        else:
            nxt = min([a for a, _ in windows if a > t] or [horizon])
            clock.advance(min(30.0, max(1e-3, nxt - t)))
        ad.tick(clock.wall())
        reports.extend(ad.take_reports())
        t = clock.monotonic()
    cap.release()
    ad.freeze()
    final = ad.finalize(None, clock.wall())
    ad.release()
    _CACHE[key] = {"reports": reports, "final": final}
    return _CACHE[key]


def morning_union(ctx=H.Context()):
    f = _run_wall(ctx)
    reports, final = f["reports"], f["final"]
    if len(reports) != 3:
        raise AssertionError(f"expected 3 buckets, got {len(reports)}")
    areas = [r.metadata["flash_union_area"] for r in reports]
    if areas != sorted(areas):
        raise AssertionError(f"union area not cumulative: {areas}")
    md = final.metadata
    if md["flash_semantics"] != "presence_or":
        raise AssertionError(f"semantics={md['flash_semantics']}")
    if md["flash_union_area"] != areas[-1] or md["flash_area"] != areas[-1]:
        raise AssertionError(
            f"morning area {md['flash_union_area']}/{md['flash_area']} != {areas[-1]}")
    if md["flash_union_area"] <= 0 or md["n_flash"] <= 0:
        raise AssertionError("morning union empty")
    if md["flash_bucket_count"] != 3:
        raise AssertionError(f"bucket_count={md['flash_bucket_count']}")
    if md["lag_frames"] != [2, 44]:
        raise AssertionError(f"lag_frames={md['lag_frames']}")
    return (f"union_area={md['flash_union_area']} n_flash={md['n_flash']} "
            f"lag={md['lag_frames']}")


def restart_equivalence(ctx=H.Context()):
    from night_lamp.session import NightSession
    s1 = NightSession("1749", _spec(True))
    _feed_session(s1, 3000)
    s1.snapshot(datetime.now(timezone.utc), force=True)
    st = s1.dump_state()
    if st is None:
        raise AssertionError("dump_state returned None")
    s2 = NightSession("1749", _spec(True))
    if not s2.load_state(st):
        raise AssertionError("load_state failed")
    if not np.array_equal(s1._flash_union, s2._flash_union):
        raise AssertionError("flash_union differs after restart")
    if s1._n_flash_buckets != s2._n_flash_buckets:
        raise AssertionError("bucket count differs after restart")
    s2.freeze()
    md = s2.finalize(None, datetime.now(timezone.utc)).metadata
    buckets = s2._n_flash_buckets
    if md["flash_union_area"] != int(s1._flash_union.sum()):
        raise AssertionError("morning area != restored union")
    if buckets != 1 or md["flash_bucket_count"] != 1:
        raise AssertionError(f"evidence count lost: {buckets}/{md['flash_bucket_count']}")
    s1.release()
    s2.release()
    return f"union_area={md['flash_union_area']} buckets={buckets}"


def continuous_no_evidence(ctx=H.Context()):
    from night_lamp.session import NightSession
    spec = _spec(False)
    s1 = NightSession("1749", spec)
    _feed_session(s1, 3000)
    st = s1.dump_state()
    s2 = NightSession("1749", spec)
    if not s2.load_state(st):
        raise AssertionError("load_state failed")
    if s2._flash_union is not None or s2._n_flash_buckets != 0:
        raise AssertionError("continuous mode fabricated evidence")
    s2.freeze()
    md = s2.finalize(None, datetime.now(timezone.utc)).metadata
    if md["flash_semantics"] != "burst_periodic":
        raise AssertionError(f"semantics={md['flash_semantics']}")
    s1.release()
    s2.release()
    return "no evidence, per-pixel fallback"


def _coverage(tick_every, frame_every, seconds=120.0):
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server.schedule import FakeClock
    spec = _spec(True)
    cap = cv.VideoCapture(str(H.ensure_video(None)))
    clock = FakeClock(datetime(2026, 9, 21, 20, 0, tzinfo=TZ))
    ad = NightAdapter(NightSession("1749", spec), spec, clock)
    seq, t, nf, nt = 0, 0.0, 0.0, 0.0
    while t < seconds:
        clock.advance(0.02)
        t += 0.02
        if t + 1e-9 >= nf:
            ok, frame = cap.read()
            if not ok:
                break
            seq += 1
            ad.on_frame(frame, clock.wall(), clock.monotonic(), frame_id=seq)
            nf += frame_every
        if t + 1e-9 >= nt:
            ad.tick(clock.wall())
            nt += tick_every
    cap.release()
    cov = ad.stats()["night_coverage"]
    ad.release()
    return cov


def tick_decoupling(ctx=H.Context()):
    fine = _coverage(0.02, 0.08)
    coarse = _coverage(0.30, 0.08)
    if fine < 0.95 or coarse >= fine:
        raise AssertionError(f"coverage fine={fine} coarse={coarse}")
    return f"coverage fine={fine:.3f} coarse={coarse:.3f}"


def test_morning_union():
    morning_union()


def test_restart_equivalence():
    restart_equivalence()


def test_continuous_no_evidence():
    continuous_no_evidence()


def checks(ctx):
    return [
        ("night.evidence_morning_union", lambda: morning_union(ctx)),
        ("night.evidence_restart", lambda: restart_equivalence(ctx)),
        ("night.evidence_continuous_sentinel", lambda: continuous_no_evidence(ctx)),
        ("night.tick_decoupling", lambda: tick_decoupling(ctx)),
    ]
