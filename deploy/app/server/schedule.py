"""运行时调度：由**注入时钟** + 配置窗口驱动的全局状态机。

状态机（contracts.ScheduleState）：
    OFF -> DAY -> NIGHT -> FROZEN -> FINALIZING -> DAY

窗口（见 tmp/wave1-interfaces.md 3.2）：
    DAY   = [day_start, day_end)           # 仅此区间内 state==DAY，day 绑定活跃
    OFF   = [day_end, night_start)         # 待机：day 算法不活跃，夜灯亦不活跃
    NIGHT = [night_start, 24:00) ∪ [00:00, night_end)   # 跨零点，逻辑不变
    FROZEN = [night_end, report_at)        # 采样停止，保留 NightSession
    report_at 当刻 -> FINALIZING

时间语义（见 tmp/wave1-interfaces.md 3.2）：
    night_start -> NIGHT_START      进入夜间采样
    night_end   -> NIGHT_FREEZE     停止采样，保留 NightSession 等 07:00
    report_at   -> NIGHT_FINALIZE   出夜灯热力图并 release
    report_interval_s -> HOURLY_SNAPSHOT（面向配置了该字段的相机）

设计约束：
  * 本模块**不认识任何具体算法名**；`is_active` 只读 `AlgorithmBinding.schedule`。
  * 一切时间判断走 `Clock`，生产用 `RealClock`，测试用 `FakeClock`。
  * `poll()` 只返回“自上次 poll 以来到期”的事件，且对同一事件实例只触发一次。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from loguru import logger

from .contracts import RuntimeConfig, ScheduleState

__all__ = [
    "Clock", "RealClock", "FakeClock",
    "RuntimeEventType", "RuntimeEvent", "Scheduler",
]

# zoneinfo 在无 tzdata 的 Windows/精简镜像上会查不到 IANA 时区；
# 这是环境能力问题，不应让调度器直接崩。用配置里出现的北京时区做固定偏移回退。
_FIXED_TZ_FALLBACK = {
    "Asia/Shanghai": timezone(timedelta(hours=8)),
    "Asia/Hong_Kong": timezone(timedelta(hours=8)),
    "UTC": timezone.utc,
}


def _resolve_tz(name: str):
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError 需要可选依赖)
        fb = _FIXED_TZ_FALLBACK.get(name)
        if fb is None:
            logger.warning("未知/不可用 IANA 时区 {!r}，回退系统本地时区", name)
            return datetime.now().astimezone().tzinfo or timezone.utc
        logger.warning("时区库缺少 {!r}，回退固定偏移 {}", name, fb)
        return fb



@runtime_checkable
class Clock(Protocol):
    def wall(self) -> datetime:
        """带时区的墙钟时间。"""
        ...

    def monotonic(self) -> float:
        """单调秒（只用于节流/耗时，不用于绝对时间）。"""
        ...


class RealClock:
    def wall(self) -> datetime:
        return datetime.now().astimezone()

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """仅测试用：显式推进墙钟与单调钟。"""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            # 明确不静默：无时区的墙钟无法做窗口判断。
            raise ValueError("FakeClock.start 必须是带时区的 datetime")
        self._wall = start
        self._mono = 0.0

    def wall(self) -> datetime:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("FakeClock 不支持倒流")
        self._wall = self._wall + timedelta(seconds=seconds)
        self._mono += seconds


class RuntimeEventType(str, Enum):
    NIGHT_START = "night_start"
    NIGHT_FREEZE = "night_freeze"
    NIGHT_FINALIZE = "night_finalize"
    HOURLY_SNAPSHOT = "hourly_snapshot"


@dataclass
class RuntimeEvent:
    type: RuntimeEventType
    at: datetime
    camera: Optional[str] = None


def _parse_hhmm(text: str) -> dtime:
    try:
        hh, mm = str(text).split(":")
        t = dtime(int(hh), int(mm))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"非法时间格式（期望 HH:MM）: {text!r}") from e
    if not (0 <= t.hour <= 23 and 0 <= t.minute <= 59):
        raise ValueError(f"非法时间: {text!r}")
    return t


def _aware(dt: datetime, tz: ZoneInfo) -> datetime:
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


class Scheduler:
    # 同一时刻多个边界时决定先后；report_at 必须先于 day_start，
    # 这样 FINALIZING 才可被观察到，day_start 不会把它抹掉。
    # day_end 置于 night_start 之前：若两者重合，night_start 应最终胜出。
    _BOUNDARIES = ("day_end", "night_start", "night_end", "report_at", "day_start")

    def __init__(self, cfg: RuntimeConfig, clock: Clock):
        self._cfg = cfg
        self._clock = clock
        self._spec = cfg.schedule
        self._tz = _resolve_tz(self._spec.timezone)
        self._state = ScheduleState.OFF
        self._last_wall: Optional[datetime] = None
        self._last_fired: dict[str, datetime] = {}
        self._snap_bucket: dict[tuple[str, str], int] = {}

    # ---------------------------------------------------------------- 查询
    def state(self) -> ScheduleState:
        return self._state

    def is_active(self, camera_id: str, algo: str) -> bool:
        """活跃性只看该绑定的 schedule 字符串；不认算法名。"""
        cam = self._camera(camera_id)
        if cam is None or not getattr(cam, "enabled", True):
            return False
        binding = (getattr(cam, "algorithms", None) or {}).get(algo)
        if binding is None:
            return False
        schedule = getattr(binding, "schedule", None)
        if schedule == "day":
            return self._state == ScheduleState.DAY
        if schedule == "night":
            return self._state == ScheduleState.NIGHT
        return False

    # ---------------------------------------------------------------- 驱动
    def poll(self) -> list[RuntimeEvent]:
        """返回自上次 poll 以来到期的事件。"""
        now = _aware(self._clock.wall(), self._tz).astimezone(self._tz)

        if self._last_wall is None:
            self._last_wall = now
            self._state = self._state_at(now)
            self._prime_snapshots(now)
            if self._state == ScheduleState.NIGHT:
                # 启动时已在夜间：通知 Runtime 加入/新建本夜 session。
                return [RuntimeEvent(RuntimeEventType.NIGHT_START, now)]
            if self._state == ScheduleState.FROZEN:
                # 采样已停、finalize 未到：补建并立即冻结，保证 07:00 能出图。
                return [
                    RuntimeEvent(RuntimeEventType.NIGHT_START, now),
                    RuntimeEvent(RuntimeEventType.NIGHT_FREEZE, now),
                ]
            return []

        prev = self._last_wall
        self._last_wall = now
        events: list[RuntimeEvent] = []

        # 上一轮已发出 NIGHT_FINALIZE，消费方已 finalize -> 回到 DAY。
        if self._state == ScheduleState.FINALIZING:
            self._state = ScheduleState.DAY

        if now > prev:
            for inst, kind in self._instants(prev, now):
                last = self._last_fired.get(kind)
                if last is not None and inst <= last:
                    continue
                self._last_fired[kind] = inst
                ev = self._apply(kind, inst)
                if ev is not None:
                    events.append(ev)

        events.extend(self._snapshots(now))
        return events

    # ---------------------------------------------------------------- 内部
    def _camera(self, camera_id: str):
        for cam in self._cfg.cameras or []:
            if getattr(cam, "id", None) == camera_id:
                return cam
        return None

    def _state_at(self, now: datetime) -> ScheduleState:
        t = now.timetz().replace(tzinfo=None)
        day_start = _parse_hhmm(self._spec.day_start)
        day_end = _parse_hhmm(self._spec.day_end)
        night_start = _parse_hhmm(self._spec.night_start)
        night_end = _parse_hhmm(self._spec.night_end)
        report_at = _parse_hhmm(self._spec.report_at)
        # NIGHT 跨零点窗口逻辑保持不变。
        if _in_night_window(t, night_start, night_end):
            return ScheduleState.NIGHT
        # 夜里采样停止后、finalize 之前。
        if _between(t, night_end, report_at):
            return ScheduleState.FROZEN
        # report_at 当刻：finalize 尚待消费。
        if t == report_at:
            return ScheduleState.FINALIZING
        # DAY 严格限定在 [day_start, day_end)；此后到 night_start 之间为待机。
        if _between(t, day_start, day_end):
            return ScheduleState.DAY
        return ScheduleState.OFF

    def _instants(self, prev: datetime, now: datetime):
        """列出 (prev, now] 内的窗口边界，按时间/优先级排序。"""
        tz = self._tz
        times = {
            "day_end": _parse_hhmm(self._spec.day_end),
            "night_start": _parse_hhmm(self._spec.night_start),
            "night_end": _parse_hhmm(self._spec.night_end),
            "report_at": _parse_hhmm(self._spec.report_at),
            "day_start": _parse_hhmm(self._spec.day_start),
        }
        order = {k: i for i, k in enumerate(self._BOUNDARIES)}
        day = (prev.astimezone(tz).date() - timedelta(days=1))
        last_day = (now.astimezone(tz).date() + timedelta(days=1))
        out = []
        while day <= last_day:
            for kind, t in times.items():
                inst = datetime.combine(day, t, tzinfo=tz)
                if prev < inst <= now:
                    out.append((inst, order[kind], kind))
            day += timedelta(days=1)
        out.sort(key=lambda x: (x[0], x[1]))
        return [(inst, kind) for inst, _o, kind in out]

    def _apply(self, kind: str, inst: datetime) -> Optional[RuntimeEvent]:
        if kind == "day_end":
            # 白昼窗口结束 -> 待机；day 绑定随 state != DAY 自动失活。
            # （正常不与 report_at 同刻；day_end 在排序中先于 report_at。）
            self._state = ScheduleState.OFF
            return None
        if kind == "night_start":
            self._state = ScheduleState.NIGHT
            return RuntimeEvent(RuntimeEventType.NIGHT_START, inst)
        if kind == "night_end":
            self._state = ScheduleState.FROZEN
            return RuntimeEvent(RuntimeEventType.NIGHT_FREEZE, inst)
        if kind == "report_at":
            self._state = ScheduleState.FINALIZING
            return RuntimeEvent(RuntimeEventType.NIGHT_FINALIZE, inst)
        if kind == "day_start":
            # 若同一时刻刚进入 FINALIZING，则不覆盖，留给下一轮过渡到 DAY。
            if self._state != ScheduleState.FINALIZING:
                self._state = ScheduleState.DAY
            return None
        return None

    # ---- 快照（HOURLY_SNAPSHOT）：通用，不认算法名 ----
    def _snapshot_targets(self):
        out = []
        for cam in self._cfg.cameras or []:
            if not getattr(cam, "enabled", True):
                continue
            for name, binding in (getattr(cam, "algorithms", None) or {}).items():
                spec = getattr(binding, "spec", None)
                interval = getattr(spec, "report_interval_s", None)
                if isinstance(interval, (int, float)) and interval > 0:
                    out.append((cam.id, name, int(interval)))
        return out

    def _day_anchor(self, now: datetime) -> datetime:
        t = _parse_hhmm(self._spec.day_start)
        anchor = datetime.combine(now.date(), t, tzinfo=self._tz)
        if now < anchor:
            anchor -= timedelta(days=1)
        return anchor

    def _bucket(self, now: datetime, interval: int) -> Optional[int]:
        elapsed = (now - self._day_anchor(now)).total_seconds()
        if elapsed < 0:
            return None
        return int(elapsed // interval)

    def _prime_snapshots(self, now: datetime) -> None:
        for cid, name, interval in self._snapshot_targets():
            b = self._bucket(now, interval)
            if b is not None:
                self._snap_bucket[(cid, name)] = b

    def _snapshots(self, now: datetime) -> list[RuntimeEvent]:
        per_cam: dict[str, datetime] = {}
        for cid, name, interval in self._snapshot_targets():
            if not self.is_active(cid, name):
                continue
            b = self._bucket(now, interval)
            if b is None or b < 1:
                continue
            key = (cid, name)
            if self._snap_bucket.get(key) == b:
                continue
            self._snap_bucket[key] = b
            per_cam.setdefault(cid, now)
        return [RuntimeEvent(RuntimeEventType.HOURLY_SNAPSHOT, at, camera=cid)
                for cid, at in per_cam.items()]


def _between(t: dtime, lo: dtime, hi: dtime) -> bool:
    """[lo, hi)（可跨零点）。"""
    if lo <= hi:
        return lo <= t < hi
    return t >= lo or t < hi


def _in_night_window(t: dtime, start: dtime, end: dtime) -> bool:
    """夜间窗口 [start, 24:00) ∪ [00:00, end)，可跨零点。"""
    if start <= end:
        return start <= t < end
    return t >= start or t < end
