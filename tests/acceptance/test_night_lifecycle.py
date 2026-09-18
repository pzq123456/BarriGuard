"""整夜生命周期验收: 19:59 -> 20:00 -> 02:59 -> 03:00 -> 06:59 -> 07:00 -> 07:01.

断言(不得放宽):
  - 03:30 起 sample_count 不再增长 (冻结后停采样)
  - 06:00 session 仍在
  - 07:00 finalize 恰好一次
  - 07:01 已 release
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402

_CACHE = {}


def _obs(ctx):
    key = ("lifecycle", ctx.video, ctx.tz)
    if key not in _CACHE:
        _CACHE[key] = H.run_lifecycle(H.ensure_video(ctx.video),
                                      tz_name=ctx.tz, nights=1)
    return _CACHE[key]


def _rec(ctx):
    return _obs(ctx).nights[-1]


def lifecycle_full(ctx=H.Context()):
    rec = H.verify_lifecycle(_obs(ctx))
    return (f"feeds={rec['feeds']} sample {rec['sample_early']}->"
            f"{rec['sample_late']} reporter={_obs(ctx).reporter_src}")


def sample_count_frozen_0330(ctx=H.Context()):
    rec = _rec(ctx)
    if rec["sample_at_0300"] != rec["sample_at_0330"]:
        raise AssertionError(
            f"sample_count changed after 03:00 freeze: "
            f"03:00={rec['sample_at_0300']} 03:30={rec['sample_at_0330']}")
    return f"frozen at {rec['sample_at_0300']}"


def session_alive_0600(ctx=H.Context()):
    rec = _rec(ctx)
    if rec["alive_at_0600"] is not True:
        raise AssertionError("NightSession not alive at 06:00")
    return "alive"


def finalize_once_0700(ctx=H.Context()):
    rec = _rec(ctx)
    if rec["finalize_events"] != 1:
        raise AssertionError(
            f"NIGHT_FINALIZE events={rec['finalize_events']} != 1")
    if rec["finalize_calls"] != 1:
        raise AssertionError(
            f"finalize calls={rec['finalize_calls']} != 1")
    return "exactly once"


def released_0701(ctx=H.Context()):
    rec = _rec(ctx)
    if rec["release_calls"] != 1 or rec["released"] is not True:
        raise AssertionError(
            f"not released by 07:01: calls={rec['release_calls']} "
            f"released={rec['released']}")
    return "released"


def test_lifecycle_full():
    lifecycle_full()


def test_sample_count_frozen_0330():
    sample_count_frozen_0330()


def test_session_alive_0600():
    session_alive_0600()


def test_finalize_once_0700():
    finalize_once_0700()


def test_released_0701():
    released_0701()


def checks(ctx):
    return [
        ("night.lifecycle_full", lambda: lifecycle_full(ctx)),
        ("night.sample_count_frozen_0330", lambda: sample_count_frozen_0330(ctx)),
        ("night.session_alive_0600", lambda: session_alive_0600(ctx)),
        ("night.finalize_once_0700", lambda: finalize_once_0700(ctx)),
        ("night.released_0701", lambda: released_0701(ctx)),
    ]
