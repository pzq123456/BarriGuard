"""生产编排验收(红线④): 直接驱动 server.worker.Runtime + CameraWorker。

test_failure_isolation 验证的是 harness 编排；本模块用 B 的 Runtime 工厂注入
(FakeClock + Spy NightAdapter + RaisingReporter)，确定性、无线程、无 RTSP，
验证 finalize / callback 失败后同一 worker 下一夜必须新建 session。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402


class _SpyFactory:
    def __init__(self, fail_finalize=False):
        self.fail_finalize = fail_finalize
        self.made = []

    def __call__(self, camera_id, spec, clock):
        from night_lamp.adapter import NightAdapter
        from night_lamp.session import NightSession

        session = NightSession(camera_id, spec)
        spy = H.Spy(NightAdapter(session, spec, clock), "NightAdapter")
        spy.session = session
        if self.fail_finalize:
            spy.fail_on("finalize")
        self.made.append(spy)
        return spy


def _run(mode):
    H.require_modules([("server.worker", "Runtime")])
    from server.schedule import FakeClock
    from server.worker import Runtime

    tz = H.resolve_tz(H.DEFAULT_TZ)
    clock = FakeClock(H.datetime(*H.REPLAY_BASE_DATE, 19, 59, tzinfo=tz))
    reporter = H.RaisingReporter("submit") if mode == "callback" else H.NullReporter()
    factory = _SpyFactory(fail_finalize=(mode == "finalize"))

    runtime = Runtime(H.build_night_only_cfg(), clock=clock, reporter=reporter,
                      night_factory=factory)
    worker = runtime.workers[H.CAMERA_ID]

    worker.start_night()
    if len(factory.made) != 1:
        raise AssertionError("Runtime NIGHT_START did not create a NightAdapter")
    first = factory.made[0]

    worker.freeze_night()
    worker.finalize_night(clock.wall())

    if first.calls.get("finalize", 0) != 1:
        raise AssertionError("finalize not called exactly once on failed night")
    if first.calls.get("release", 0) != 1:
        raise AssertionError("failed night did not release its session")

    worker.start_night()
    if len(factory.made) != 2:
        raise AssertionError(
            "next night reused the stale night slot (no new NightAdapter)")
    second = factory.made[1]
    if second is first or getattr(second, "session", None) is first.session:
        raise AssertionError("next night reused the previous session object")
    return f"{mode}: released={first.calls} -> new {type(second.session).__name__}"


def test_runtime_finalize_failure_new_session():
    _run("finalize")


def test_runtime_callback_failure_new_session():
    _run("callback")


def checks(ctx):
    return [
        ("runtime.finalize_failure_new_session",
         test_runtime_finalize_failure_new_session),
        ("runtime.callback_failure_new_session",
         test_runtime_callback_failure_new_session),
    ]
