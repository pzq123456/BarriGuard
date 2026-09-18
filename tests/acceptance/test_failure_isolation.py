"""夜隔离验收(红线④): finalize / callback 失败后，下一夜必须是全新 session。

断言:
  - 失败夜记录到错误(未被吞掉)
  - 失败夜的 session 仍被 release
  - 下一夜创建新 session，且与旧 session 不是同一对象
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


def _obs(ctx, mode):
    key = (mode, ctx.video, ctx.tz)
    if key not in _CACHE:
        kw = {"finalize_fail": mode == "finalize",
              "callback_fail": mode == "callback"}
        obs = H.run_lifecycle(H.ensure_video(ctx.video), tz_name=ctx.tz,
                              nights=2, **kw)
        _CACHE[key] = obs
    return _CACHE[key]


def _check(ctx, mode):
    obs = _obs(ctx, mode)
    info = H.verify_failure_isolation(obs)
    return f"{mode}: {info['night1_errors']} -> new {info['new_session']}"


def test_finalize_failure_new_session():
    _check(H.Context(), "finalize")


def test_callback_failure_new_session():
    _check(H.Context(), "callback")


def checks(ctx):
    return [
        ("isolation.finalize_failure", lambda: _check(ctx, "finalize")),
        ("isolation.callback_failure", lambda: _check(ctx, "callback")),
    ]
