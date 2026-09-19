"""Overnight burst 验收: 每小时 burst 出图, 空间累计跨 burst, 时间态逐 burst 重置.

用生产 ``NightSession`` + ``NightAdapter`` + ``FakeClock`` 驱动加速 cadence，
不等待真实小时；不改变算法语义。
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "app"
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402

TZ = timezone(timedelta(hours=8))
_CACHE = {}


def _run(ctx):
    key = ("overnight", ctx.video)
    if key in _CACHE:
        return _CACHE[key]

    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server.contracts import (AlignmentSpec, BaselineSpec, MemorySpec,
                                  NightLampSpec, OvernightSpec, PeriodicitySpec,
                                  SamplingSpec)
    from server.schedule import FakeClock

    spec = NightLampSpec(
        sampling=SamplingSpec(interval_ms=160),
        memory=MemorySpec(max_candidates=200, series_cap=None),
        baseline=BaselineSpec(frames=60, warmup_s=600),
        periodicity=PeriodicitySpec(onset_min=50),
        alignment=AlignmentSpec(),
        overnight=OvernightSpec(enabled=True, cadence_minutes=1,
                                burst_seconds=20),
    )
    cadence_min, burst_s, bursts = 1, 20, 2
    step = spec.sampling.interval_ms / 1000.0 * 1.05
    reader = H.Mp4Reader(H.ensure_video(ctx.video), loop=True).start()
    clock = FakeClock(datetime(2026, 9, 18, 20, 0, tzinfo=TZ))
    session = NightSession(H.CAMERA_ID, spec)
    adapter = NightAdapter(session, spec, clock)

    series_ids, series_refs = [], []
    orig_begin = session.begin_burst

    def begin(ts):
        series_refs.append(session._series)   # hold a ref so id() cannot be reused
        series_ids.append(id(session._series))
        return orig_begin(ts)

    session.begin_burst = begin

    reports, cum_obs = [], []
    t, total = 0.0, cadence_min * 60 * bursts
    try:
        while t < total:
            frame = reader.read()
            if frame is None:
                break
            clock.advance(step)
            t += step
            adapter.on_frame(frame, clock.wall(), clock.monotonic())
            for rep in adapter.take_reports():
                reports.append(rep)
                cum_obs.append(rep.metadata["sampling"]["actual_samples"])
        adapter.freeze()
        final = adapter.finalize(None, clock.wall())
    finally:
        adapter.release()
        reader.stop()

    facts = {
        "reports": reports,
        "cum_obs": cum_obs,
        "series_reset": len(series_refs) == bursts == len(set(series_ids)),
        "final": final,
        "stats": adapter.stats(),
        "released": session._released,
    }
    _CACHE[key] = facts
    return facts


def burst_heatmaps(ctx=H.Context()):
    f = _run(ctx)
    if len(f["reports"]) != 2:
        raise AssertionError(f"expected 2 burst heatmaps, got {len(f['reports'])}")
    for rep in f["reports"]:
        if rep.report_type != "night_heatmap_burst":
            raise AssertionError(f"report_type={rep.report_type}")
        if not rep.image_jpeg:
            raise AssertionError("burst heatmap missing JPEG")
    return f"2 bursts, jpegs={[len(r.image_jpeg) for r in f['reports']]}"


def spatial_persists(ctx=H.Context()):
    f = _run(ctx)
    obs = f["cum_obs"]
    if len(obs) != 2 or obs[1] <= obs[0]:
        raise AssertionError(f"spatial accumulator not cumulative: {obs}")
    return f"cumulative observations {obs}"


def temporal_resets(ctx=H.Context()):
    f = _run(ctx)
    if not f["series_reset"]:
        raise AssertionError("temporal state was not reset per burst")
    return "temporal series replaced at every burst"


def drops_counted(ctx=H.Context()):
    f = _run(ctx)
    st = f["stats"]
    if st["dropped_burst"] <= 0:
        raise AssertionError("out-of-burst frames not counted as dropped")
    if st["observations_dropped"] != st["dropped_burst"] + st["dropped_interval"]:
        raise AssertionError(f"drop counters inconsistent: {st}")
    return f"received={st['observations_received']} dropped={st['observations_dropped']}"


def morning_finalize(ctx=H.Context()):
    f = _run(ctx)
    if f["final"] is None or f["final"].report_type != "night_heatmap":
        raise AssertionError("morning finalize did not produce night_heatmap")
    if not f["released"]:
        raise AssertionError("session not released after finalize")
    return f"final status={f['final'].status} released={f['released']}"


def checks(ctx):
    return [
        ("overnight.burst_heatmaps", lambda: burst_heatmaps(ctx)),
        ("overnight.spatial_persists", lambda: spatial_persists(ctx)),
        ("overnight.temporal_resets", lambda: temporal_resets(ctx)),
        ("overnight.drops_counted", lambda: drops_counted(ctx)),
        ("overnight.morning_finalize", lambda: morning_finalize(ctx)),
    ]
