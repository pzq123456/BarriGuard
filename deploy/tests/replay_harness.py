"""Wave 1 Agent F: 回放 / 验收 harness (tests/replay_harness.py).

边界
----
- source 换成 MP4: ``Mp4Reader`` 提供与 ``server/source.py::Reader`` 相同的公开
  逐帧接口 (start/stop/read/burst_next)，用于确定性回放；不复制 nightly_map /
  NightSession 逻辑。
- 生命周期由生产 ``server.schedule`` 的 ``FakeClock`` + ``Scheduler`` 驱动；
  夜灯累积/冻结/finalize 全部委托生产 ``night_lamp.session.NightSession`` 与
  ``night_lamp.adapter.NightAdapter``；上报委托生产 ``server.reporting.Reporter``。
- 生产模块可能尚未实现: 缺失时该检查标记 ``UNVERIFIED`` 并列出依赖，
  绝不为通过而放宽断言。

检查状态
--------
PASS       断言全部成立
FAIL       模块已实现但契约/行为不符 (必须修生产代码)
UNVERIFIED 依赖模块尚未落地 / 环境不满足 (明确列出依赖)

退出码: 0=全 PASS; 1=有 FAIL; 2=无 FAIL 但有 UNVERIFIED (不得当作绿灯)

用法
----
    python tests/replay_harness.py --selftest
    python tests/replay_harness.py --video tmp/1749_202609040100.mp4
    python tests/replay_harness.py --acceptance
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

_TESTS = Path(__file__).resolve().parent
ROOT = _TESTS.parent / "app"
TESTS = _TESTS
REPO_ROOT = _TESTS.parent.parent
for _p in (str(ROOT), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NIGHT_START = "night_start"
NIGHT_FREEZE = "night_freeze"
NIGHT_FINALIZE = "night_finalize"

DAY_ALGO = "water_gap"
NIGHT_ALGO = "night_lamp"
CAMERA_ID = "1749"

READER_SURFACE = ("start", "stop", "read", "burst_next")
DEFAULT_TZ = "Asia/Shanghai"
REPLAY_BASE_DATE = (2026, 9, 18)


class Unverified(Exception):
    """依赖缺失或环境不满足，无法判定。"""


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


@dataclass
class Context:
    video: Optional[str] = None
    tz: str = DEFAULT_TZ


@dataclass
class LifecycleObs:
    nights: list = field(default_factory=list)
    events: list = field(default_factory=list)
    reporter_src: str = "unknown"
    tz_used: str = ""


class InjectedFailure(RuntimeError):
    pass


class Spy:
    """记录方法调用次数并原样委托；不含任何业务逻辑。"""

    def __init__(self, target, name="target"):
        self._target = target
        self.name = name
        self.calls = {}
        self.inject = {}

    def fail_on(self, method):
        self.inject[method] = True

    def __getattr__(self, item):
        attr = getattr(self._target, item)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            self.calls[item] = self.calls.get(item, 0) + 1
            if self.inject.get(item):
                raise InjectedFailure(f"injected failure in {item}")
            return attr(*args, **kwargs)

        return wrapper


class NullReporter:
    def __init__(self):
        self.reports = []
        self.events = []

    def submit(self, report):
        self.reports.append(report)

    def submit_event(self, event):
        self.events.append(event)

    def close(self):
        pass


class RaisingReporter(NullReporter):
    def __init__(self, point="submit"):
        super().__init__()
        self.point = point

    def submit(self, report):
        super().submit(report)
        raise InjectedFailure(f"injected callback failure at {self.point}")


class Mp4Reader:
    """确定性 MP4 逐帧源，公开接口与生产 ``server.source.Reader`` 对齐。

    与生产 Reader 的差异（刻意、仅限测试）：顺序读帧而非只留最新帧；
    ``start()`` 不拉后台线程。逐帧语义等价，用于可复现回放。
    """

    def __init__(self, path, fps=None, loop=True):
        self.path = str(path)
        self._cap = cv.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self._fps = float(fps or self._cap.get(cv.CAP_PROP_FPS) or 25.0)
        self._total = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))
        self._lock = threading.Lock()
        self._loop = loop
        self._stop = threading.Event()
        self._seq = 0
        self._read_n = 0

    def start(self):
        self._stop.clear()
        return self

    def stop(self):
        self._stop.set()
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def read(self):
        if self._stop.is_set():
            return None
        with self._lock:
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
            if not ok:
                if not self._loop:
                    return None
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    return None
            self._seq += 1
            self._read_n += 1
            return frame.copy()

    def burst_next(self, n, timeout_s=30.0):
        out = []
        for _ in range(n):
            frame = self.read()
            if frame is None:
                raise TimeoutError(f"burst_next eof {len(out)}/{n}")
            out.append(frame)
        return out

    def fps(self):
        return self._fps

    def total(self):
        return self._total

    def frames_read(self):
        return self._read_n


def resolve_tz(name: str):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return timezone(timedelta(hours=8), name)


def ensure_video(path=None) -> Path:
    if path:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"video not found: {path}")
    env = os.environ.get("BARRIGUARD_REPLAY_VIDEO")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"BARRIGUARD_REPLAY_VIDEO not found: {env}")
    cands = sorted((REPO_ROOT / "tmp").glob("*.mp4"))
    if cands:
        return cands[0]
    return synth_video()


def synth_video() -> Path:
    out = REPO_ROOT / "tmp" / "replay_harness_synth.mp4"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = getattr(cv, "VideoWriter_fourcc", None) or cv.VideoWriter.fourcc
    vw = cv.VideoWriter(str(out), fourcc(*"mp4v"), 10.0, (64, 48))
    if not vw.isOpened():
        raise RuntimeError("cannot create synthetic mp4")
    for i in range(20):
        vw.write(np.full((48, 64, 3), (i * 10) % 255, np.uint8))
    vw.release()
    return out


def require_modules(resources):
    missing = []
    for module, attr in resources:
        try:
            mod = importlib.import_module(module)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{module} [{type(e).__name__}]")
            continue
        if attr and not hasattr(mod, attr):
            missing.append(f"{module}.{attr}")
    if missing:
        raise Unverified("missing dependencies: " + ", ".join(missing))


def _load_attr(module, qualname):
    try:
        target = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        raise Unverified(f"{module} [{type(e).__name__}: {e}]")
    for part in qualname.split("."):
        if not hasattr(target, part):
            raise Unverified(f"{module}.{qualname} not implemented")
        target = getattr(target, part)
    return target


def build_cfg(tz_name=DEFAULT_TZ):
    from server.contracts import (AlgorithmBinding, CallbackSpec, CameraSpec,
                                  NightLampSpec, RuntimeConfig, ScheduleSpec,
                                  WaterGapSpec)

    cam = CameraSpec(
        id=CAMERA_ID,
        name=f"replay-{CAMERA_ID}",
        rtsp_url="file://replay.mp4",
        algorithms={
            DAY_ALGO: AlgorithmBinding(name=DAY_ALGO, schedule="day",
                                       spec=WaterGapSpec(schedule="day")),
            NIGHT_ALGO: AlgorithmBinding(name=NIGHT_ALGO, schedule="night",
                                         spec=NightLampSpec(schedule="night")),
        },
    )
    return RuntimeConfig(
        schedule=ScheduleSpec(timezone=tz_name),
        callback=CallbackSpec(url="http://127.0.0.1:9/replay", enabled=True),
        cameras=[cam],
    )


def build_night_only_cfg(tz_name=DEFAULT_TZ):
    from server.contracts import (AlgorithmBinding, CallbackSpec, CameraSpec,
                                  NightLampSpec, RuntimeConfig, ScheduleSpec)

    cam = CameraSpec(
        id=CAMERA_ID,
        name=f"replay-{CAMERA_ID}",
        rtsp_url="file://replay.mp4",
        algorithms={
            NIGHT_ALGO: AlgorithmBinding(name=NIGHT_ALGO, schedule="night",
                                         spec=NightLampSpec(schedule="night")),
        },
    )
    return RuntimeConfig(
        schedule=ScheduleSpec(timezone=tz_name),
        callback=CallbackSpec(url="http://127.0.0.1:9/replay", enabled=True),
        cameras=[cam],
    )


def _kind(event):
    raw = getattr(event, "type", None)
    return getattr(raw, "value", raw)


def _sample_count(session):
    if session is None or not hasattr(session, "sample_count"):
        raise Unverified("NightSession.sample_count unavailable")
    return int(session.sample_count)


def _make_reporter(cfg, callback_fail):
    if callback_fail:
        return RaisingReporter("submit"), "raising"
    try:
        from server.reporting import Reporter

        return Reporter(cfg), "production"
    except Exception:  # noqa: BLE001
        return NullReporter(), "harness-fallback"


def _blank_record(index):
    return {
        "index": index,
        "adapter": None,
        "session": None,
        "feeds": 0,
        "sample_early": None,
        "sample_late": None,
        "sample_at_0300": None,
        "sample_at_0330": None,
        "alive_at_0600": None,
        "alive_at_0659": None,
        "finalize_events_0659": 0,
        "finalize_events": 0,
        "finalize_calls": 0,
        "release_calls": 0,
        "released": False,
        "report": None,
        "errors": [],
        "early_start": False,
    }


def run_lifecycle(video, *, tz_name=DEFAULT_TZ, nights=1,
                  finalize_fail=False, callback_fail=False,
                  feed_step_min=10) -> LifecycleObs:
    """用生产 FakeClock + Scheduler 驱动整夜生命周期。

    nights=2 时跨两夜；finalize_fail/callback_fail 注入故障验证夜隔离。
    """
    require_modules([
        ("server.schedule", "Scheduler"),
        ("server.schedule", "FakeClock"),
        ("night_lamp.session", "NightSession"),
        ("night_lamp.adapter", "NightAdapter"),
    ])
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server.schedule import FakeClock, Scheduler

    tz = resolve_tz(tz_name)
    cfg = build_cfg(tz_name)
    cam = cfg.cameras[0]
    night_spec = cam.algorithms[NIGHT_ALGO].spec

    reporter, reporter_src = _make_reporter(cfg, callback_fail)
    reader = Mp4Reader(video, loop=True).start()

    start = datetime(*REPLAY_BASE_DATE, 19, 59, tzinfo=tz)
    records = []
    events = []

    try:
        clock = FakeClock(start)
    except Exception as e:  # noqa: BLE001
        reader.stop()
        raise Unverified(f"FakeClock init failed: {type(e).__name__}: {e}")

    try:
        scheduler = Scheduler(cfg, clock)
    except Exception as e:  # noqa: BLE001
        reader.stop()
        if "ZoneInfo" in type(e).__name__ or "time zone" in str(e).lower():
            raise Unverified(
                f"Scheduler needs zoneinfo data (install tzdata): {e}")
        raise

    def advance(dt):
        delta = (dt - clock.wall()).total_seconds()
        if delta > 0:
            clock.advance(delta)

    def state_value():
        try:
            st = scheduler.state()
        except Exception:  # noqa: BLE001
            return "?"
        return str(getattr(st, "value", st))

    def start_night(rec):
        session = NightSession(cam.id, night_spec)
        adapter = NightAdapter(session, night_spec, clock)
        spy = Spy(adapter, "NightAdapter")
        if finalize_fail:
            spy.fail_on("finalize")
        rec["session"] = session
        rec["adapter"] = spy

    def do_finalize(rec, dt):
        base = reader.read()
        try:
            report = rec["adapter"].finalize(base, dt)
            rec["report"] = report
            reporter.submit(report)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(("finalize", f"{type(e).__name__}: {e}"))
        finally:
            try:
                rec["adapter"].release()
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(("release", f"{type(e).__name__}: {e}"))
            rec["finalize_calls"] = rec["adapter"].calls.get("finalize", 0)
            rec["release_calls"] = rec["adapter"].calls.get("release", 0)
            rec["released"] = True

    def handle(evs, dt, rec):
        for ev in evs:
            kind = _kind(ev)
            events.append((kind, dt.isoformat()))
            if kind == NIGHT_START:
                start_night(rec)
            elif kind == NIGHT_FREEZE:
                if rec["adapter"] is not None:
                    rec["adapter"].freeze()
            elif kind == NIGHT_FINALIZE:
                rec["finalize_events"] += 1
                if rec["finalize_events"] == 1 and rec["adapter"] is not None:
                    do_finalize(rec, dt)

    def poll(dt, rec):
        advance(dt)
        evs = scheduler.poll()
        if rec is not None:
            handle(evs, dt, rec)
        return evs

    try:
        base = datetime(*REPLAY_BASE_DATE, tzinfo=tz)
        for n in range(nights):
            day = base + timedelta(days=n)
            night_start = day.replace(hour=20, minute=0)
            freeze_at = night_start + timedelta(hours=7)
            report_at = night_start + timedelta(hours=11)
            rec = _blank_record(n)
            records.append(rec)

            pre = poll(night_start - timedelta(minutes=1), None)
            rec["early_start"] = any(_kind(e) == NIGHT_START for e in pre)

            poll(night_start, rec)
            if rec["adapter"] is None:
                continue

            minutes = 0
            while True:
                t = night_start + timedelta(minutes=minutes)
                if t >= freeze_at:
                    break
                advance(t)
                if state_value() == "night":
                    rec["adapter"].on_frame(reader.read(), t,
                                            (t - start).total_seconds())
                    rec["feeds"] += 1
                if minutes == feed_step_min:
                    rec["sample_early"] = _sample_count(rec["session"])
                minutes += feed_step_min

            last_t = freeze_at - timedelta(minutes=1)
            advance(last_t)
            if state_value() == "night":
                rec["adapter"].on_frame(reader.read(), last_t,
                                        (last_t - start).total_seconds())
                rec["feeds"] += 1
            rec["sample_late"] = _sample_count(rec["session"])

            poll(freeze_at, rec)
            rec["sample_at_0300"] = _sample_count(rec["session"])

            poll(freeze_at + timedelta(minutes=30), rec)
            rec["sample_at_0330"] = _sample_count(rec["session"])

            poll(night_start + timedelta(hours=10), rec)
            rec["alive_at_0600"] = rec["adapter"] is not None and not rec["released"]

            poll(report_at - timedelta(minutes=1), rec)
            rec["finalize_events_0659"] = rec["finalize_events"]
            rec["alive_at_0659"] = rec["adapter"] is not None and not rec["released"]

            poll(report_at, rec)
            if rec["adapter"] is not None:
                rec["finalize_calls"] = rec["adapter"].calls.get("finalize", 0)
                rec["release_calls"] = rec["adapter"].calls.get("release", 0)

            poll(report_at + timedelta(minutes=1), rec)
    finally:
        try:
            reporter.close()
        except Exception:  # noqa: BLE001
            pass
        reader.stop()

    return LifecycleObs(nights=records, events=events,
                        reporter_src=reporter_src, tz_used=str(tz))


def verify_lifecycle(obs, expect_report=True):
    if not obs.nights:
        raise AssertionError("no night record produced")
    rec = obs.nights[-1]
    if rec["early_start"]:
        raise AssertionError("NIGHT_START fired before 20:00")
    if rec["adapter"] is None:
        raise AssertionError("NIGHT_START did not create a NightSession")
    if rec["feeds"] <= 0:
        raise AssertionError("no frames fed during NIGHT (scheduler state?)")
    if rec["sample_early"] is None or rec["sample_late"] is None:
        raise Unverified("sample_count unavailable during night")

    if rec["sample_late"] <= rec["sample_early"]:
        raise AssertionError(
            f"sample_count did not grow during night: early={rec['sample_early']} "
            f"late={rec['sample_late']}")
    if rec["sample_at_0300"] != rec["sample_at_0330"]:
        raise AssertionError(
            "sample_count grew after 03:00 freeze: "
            f"03:00={rec['sample_at_0300']} 03:30={rec['sample_at_0330']}")
    if rec["sample_at_0330"] < rec["sample_late"]:
        raise AssertionError("sample_count below pre-freeze value after freeze")
    if rec["alive_at_0600"] is not True:
        raise AssertionError("NightSession not alive at 06:00")
    if rec["alive_at_0659"] is not True:
        raise AssertionError("NightSession not alive at 06:59 (premature finalize?)")
    if rec["finalize_events_0659"] != 0:
        raise AssertionError(
            f"NIGHT_FINALIZE fired before 07:00: {rec['finalize_events_0659']}")
    if rec["finalize_events"] != 1:
        raise AssertionError(
            f"NIGHT_FINALIZE must fire exactly once, got {rec['finalize_events']}")
    if rec["finalize_calls"] != 1:
        raise AssertionError(
            f"finalize must be called exactly once, got {rec['finalize_calls']}")
    if rec["release_calls"] != 1:
        raise AssertionError(
            f"release must be called exactly once, got {rec['release_calls']}")
    if rec["released"] is not True:
        raise AssertionError("session not released after finalize")
    if expect_report and rec["report"] is None:
        raise AssertionError("finalize produced no Report")
    return rec


def verify_failure_isolation(obs):
    if len(obs.nights) < 2:
        raise AssertionError("need two nights to verify isolation")
    n1, n2 = obs.nights[0], obs.nights[1]
    if not n1["errors"]:
        raise AssertionError("injected finalize/callback failure was swallowed")
    if n1["release_calls"] != 1:
        raise AssertionError("failed night did not release its session")
    if n2["session"] is None:
        raise AssertionError("next night did not create a session")
    if n2["session"] is n1["session"]:
        raise AssertionError("next night reused previous night's session")
    return {"night1_errors": n1["errors"], "new_session": type(n2["session"]).__name__}


def check_reader_surface():
    from server.source import Reader

    missing = [m for m in READER_SURFACE
               if not hasattr(Mp4Reader, m) or not hasattr(Reader, m)]
    if missing:
        raise AssertionError(f"Mp4Reader/Reader missing surface: {missing}")
    return f"surface={READER_SURFACE}"


def check_mp4_reader(video=None):
    path = ensure_video(video)
    reader = Mp4Reader(path, loop=True).start()
    try:
        frames = reader.burst_next(3)
        if len(frames) != 3:
            raise AssertionError("burst_next did not return 3 frames")
        shapes = {f.shape for f in frames}
        if len(shapes) != 1:
            raise AssertionError(f"frame shapes inconsistent: {shapes}")
        if not all(f.dtype == np.uint8 for f in frames[:1]):
            raise AssertionError("frames not uint8")
        if reader.frames_read() < 3:
            raise AssertionError("frames_read counter not advancing")
        return f"{path.name} fps={reader.fps():.1f} total={reader.total()} shape={shapes.pop()}"
    finally:
        reader.stop()


FROZEN_SIGNATURES = {
    "server.schedule": [
        ("Scheduler.__init__", ["self", "cfg", "clock"]),
        ("Scheduler.state", ["self"]),
        ("Scheduler.is_active", ["self", "camera_id", "algo"]),
        ("Scheduler.poll", ["self"]),
        ("FakeClock.__init__", ["self", "start"]),
        ("FakeClock.advance", ["self", "seconds"]),
    ],
    "server.agents.day": [
        ("DayRunner.__init__", ["self", "camera", "spec", "algo"]),
        ("DayRunner.on_frame", ["self", "frame_bgr", "ts_mono"]),
        ("DayRunner.snapshot", ["self", "ts_wall"]),
    ],
    "night_lamp.session": [
        ("NightSession.__init__", ["self", "camera_id", "spec"]),
        ("NightSession.accumulate", ["self", "frame_bgr", "ts_mono"]),
        ("NightSession.freeze", ["self"]),
        ("NightSession.finalize", ["self", "base_frame_bgr", "ts_wall"]),
        ("NightSession.release", ["self"]),
    ],
    "night_lamp.adapter": [
        ("NightAdapter.__init__", ["self", "session", "spec", "clock"]),
        ("NightAdapter.on_frame", ["self", "frame_bgr", "ts_wall", "ts_mono"]),
        ("NightAdapter.freeze", ["self"]),
        ("NightAdapter.finalize", ["self", "base_frame_bgr", "ts_wall"]),
        ("NightAdapter.release", ["self"]),
    ],
    "server.reporting": [
        ("Reporter.__init__", ["self", "cfg"]),
        ("Reporter.submit", ["self", "report"]),
        ("Reporter.submit_event", ["self", "event", "image_jpeg"]),
        ("Reporter.close", ["self"]),
    ],
}


def check_signatures(module):
    spec = FROZEN_SIGNATURES[module]
    checked = []
    for qualname, expected in spec:
        obj = _load_attr(module, qualname)
        got = list(inspect.signature(obj).parameters)
        if got != expected:
            raise AssertionError(
                f"{module}.{qualname} params {got} != frozen {expected}")
        checked.append(qualname)
    return f"{len(checked)} signatures match"


def check_night_sample_count_property():
    cls = _load_attr("night_lamp.session", "NightSession")
    prop = cls.__dict__.get("sample_count")
    if not isinstance(prop, property):
        raise AssertionError("NightSession.sample_count must be a property")
    return "sample_count is property"


def run_check(name, fn) -> Result:
    try:
        detail = fn()
        return Result(name, "PASS", detail or "")
    except Unverified as e:
        return Result(name, "UNVERIFIED", str(e))
    except AssertionError as e:
        return Result(name, "FAIL", str(e))
    except Exception as e:  # noqa: BLE001
        return Result(name, "FAIL", f"{type(e).__name__}: {e}")


def run_checks(items) -> list:
    return [run_check(name, fn) for name, fn in items]


def format_results(results) -> str:
    width = max((len(r.name) for r in results), default=0)
    lines = []
    for r in results:
        lines.append(f"[{r.status:10}] {r.name:<{width}}  {r.detail}")
    counts = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    lines.append("")
    lines.append(f"SUMMARY: {summary}")
    unverified = [r for r in results if r.status == "UNVERIFIED"]
    if unverified:
        lines.append("UNVERIFIED dependencies (not green):")
        for r in unverified:
            lines.append(f"  - {r.name}: {r.detail}")
    return "\n".join(lines)


def builtin_checks(ctx):
    def lifecycle():
        obs = run_lifecycle(ensure_video(ctx.video), tz_name=ctx.tz, nights=1)
        verify_lifecycle(obs)
        return f"reporter={obs.reporter_src}"

    return [
        ("source.reader_surface", check_reader_surface),
        ("source.mp4_reader", lambda: check_mp4_reader(ctx.video)),
        ("schedule.signatures", lambda: check_signatures("server.schedule")),
        ("day.signatures", lambda: check_signatures("server.agents.day")),
        ("night.session.signatures",
         lambda: check_signatures("night_lamp.session")),
        ("night.adapter.signatures",
         lambda: check_signatures("night_lamp.adapter")),
        ("night.sample_count.property", check_night_sample_count_property),
        ("reporting.signatures", lambda: check_signatures("server.reporting")),
        ("lifecycle.full", lifecycle),
    ]


def selftest_checks(ctx):
    return [
        ("source.reader_surface", check_reader_surface),
        ("source.mp4_reader", lambda: check_mp4_reader(ctx.video)),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description="BarriGuard Wave1 replay/acceptance harness")
    ap.add_argument("--video", help="MP4 path (default: first tmp/*.mp4 or synthetic)")
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--selftest", action="store_true",
                    help="only harness plumbing (no production night modules)")
    ap.add_argument("--acceptance", action="store_true",
                    help="run tests/acceptance/*")
    args = ap.parse_args(argv)

    ctx = Context(video=args.video, tz=args.tz)
    if args.selftest:
        results = run_checks(selftest_checks(ctx))
    elif args.acceptance:
        import acceptance

        results = acceptance.run_all(ctx)
    else:
        results = run_checks(builtin_checks(ctx))

    print(format_results(results))

    statuses = {r.status for r in results}
    if "FAIL" in statuses:
        return 1
    if "UNVERIFIED" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
