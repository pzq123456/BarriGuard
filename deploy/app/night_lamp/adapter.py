"""NightAdapter: interval-gated feed of a NightSession (Wave 1, Agent D).

The adapter owns the ``sampling.interval_ms`` cadence using the injected
``clock`` (``schedule.Clock`` protocol); the session owns accumulation and
knows nothing about time gating.  One adapter per camera per night.

Overnight burst mode (``spec.overnight.enabled``)
------------------------------------------------
Instead of one continuous 7 h session, the adapter runs a fixed hourly cadence:

    |-- burst_seconds ON --|------ cadence - burst OFF ------|

During a burst it feeds the session normally; at the burst end it asks the
session for a cumulative heatmap snapshot and queues it (``take_reports``).
Spatial accumulators persist in the session across bursts; only the temporal
tracking is reset, by ``NightSession.begin_burst``.  ``enabled=False`` keeps
the original continuous behaviour.
"""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta

import numpy as np
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.contracts import NightLampSpec  # noqa: E402

from night_lamp import report as R  # noqa: E402
from night_lamp.persist import night_key  # noqa: E402
from night_lamp.session import NightSession  # noqa: E402

_SAVE_INTERVAL_S = 600.0


class NightAdapter:
    """Gate frames to the session at most once per ``interval_ms``."""

    def __init__(self, session: NightSession, spec: NightLampSpec, clock):
        self._session = session
        self._spec = spec
        self._clock = clock
        self._interval_s = max(int(spec.sampling.interval_ms), 1) / 1000.0
        self._last_emit = None
        self._frozen = False
        self._released = False
        self._emitted = 0

        ov = getattr(spec, "overnight", None)
        self._overnight = bool(getattr(ov, "enabled", False))
        self._cadence_s = max(int(getattr(ov, "cadence_minutes", 60)), 1) * 60.0
        self._burst_s = max(int(getattr(ov, "burst_seconds", 120)), 1)
        self._night_start_mono = None
        self._burst_active = False
        self._bursts_done = 0
        self._pending: list = []
        self._pending_lock = threading.Lock()
        self._received = 0
        self._dropped_interval = 0
        self._dropped_burst = 0

        # 墙钟分桶（anchor=wall）：由独立 tick 驱动，on_frame 只喂数据。
        self._anchor = str(getattr(ov, "anchor", "stream") or "stream")
        self._wall = (self._anchor == "wall")
        self._bucket_id = None
        self._window_open = False
        self._window_start = None
        self._window_end = None
        self._win_received = 0
        self._win_emitted = 0
        self._last_frame_id = None
        self._dropped_repeat = 0
        self._last_emitted_bucket = None
        self._night_scheduled = 0
        self._night_emitted = 0
        # 固定网格重采样：on_frame 只登记最新帧，采样由 tick 按 interval 网格完成。
        self._latest_frame = None
        self._latest_id = None
        self._grid_next = None

        self._store = None
        self._archive = None
        self._night_key = None
        self._last_save_mono = None
        self._save_interval_s = _SAVE_INTERVAL_S
        self._restored = False
        self._camera_id = getattr(session, "camera_id", None)

    @property
    def emitted(self) -> int:
        return self._emitted

    def attach_persistence(self, store) -> None:
        """绑定夜间状态 store；store 为空或未启用时整体 no-op。"""
        self._store = store if getattr(store, "enabled", False) else None

    def attach_archive(self, archive) -> None:
        """绑定 L1 归档；archive 为空或未启用时 no-op。"""
        self._archive = archive if getattr(archive, "enabled", False) else None

    def restore(self, ts_wall=None) -> bool:
        """按本夜 key 恢复累计状态；成功则跳过预热继续累积。"""
        if self._store is None:
            return False
        self._night_key = night_key(ts_wall or self._clock.wall())
        state = self._store.load(self._camera_id, self._night_key)
        if state is None:
            return False
        try:
            lb = state.get("last_bucket")
            if lb is not None:
                self._last_emitted_bucket = str(np.asarray(lb).reshape(-1)[0])
        except Exception:
            pass
        ok = self._session.load_state(state)
        self._restored = bool(ok)
        if ok:
            logger.info("[night] 恢复 {} {} 累计: {} samples",
                        self._camera_id, self._night_key,
                        self._session.sample_count)
        return self._restored

    def save_state(self, ts_wall=None) -> bool:
        """把当前累计原子落盘；无 store 或未预热则 no-op。"""
        self._archive_now(ts_wall)
        if self._store is None:
            return False
        if self._night_key is None:
            self._night_key = night_key(ts_wall or self._clock.wall())
        state = self._session.dump_state()
        if state is None:
            return False
        state["last_bucket"] = np.array(self._last_emitted_bucket or "")
        return self._store.save(self._camera_id, self._night_key, state) is not None

    def _archive_now(self, ts_wall=None) -> bool:
        """L1 归档热力图核心数组；出图后仍保留，供离线复算。"""
        if self._archive is None:
            return False
        if self._night_key is None:
            self._night_key = night_key(ts_wall or self._clock.wall())
        state = self._session.dump_state()
        if state is None:
            return False
        now = ts_wall or self._clock.wall()
        manifest = {
            "camera_id": self._camera_id,
            "night": self._night_key,
            "updated_at": now.isoformat() if hasattr(now, "isoformat") else str(now),
            "n_seen": int(self._session.sample_count),
            "candidate_overflow": bool(self._session.candidate_overflow),
            "restored": self._restored,
            "bursts_done": self._bursts_done,
        }
        return self._archive.save(
            self._camera_id, self._night_key, state, manifest) is not None

    def on_frame(self, frame_bgr, ts_wall, ts_mono, frame_id=None) -> None:
        if self._released or self._frozen:
            return
        now = self._clock.monotonic()
        self._received += 1
        if self._overnight:
            if self._wall:
                if not self._window_open:
                    self._dropped_burst += 1
                    return
                # 只登记最新帧；真正的采样落到固定网格（见 tick），
                # 从而把网络到达抖动重采样成均匀时间轴。
                self._win_received += 1
                self._latest_frame = frame_bgr
                self._latest_id = frame_id
                return
            else:
                if self._night_start_mono is None:
                    self._night_start_mono = now
                elapsed = now - self._night_start_mono
                in_burst = (elapsed % self._cadence_s) < self._burst_s
                if in_burst and not self._burst_active:
                    self._burst_active = True
                    self._session.begin_burst(ts_wall)
                elif not in_burst and self._burst_active:
                    self._end_burst(ts_wall)
                if not in_burst:
                    self._dropped_burst += 1
                    return
        else:
            self._maybe_periodic_save(now)

        if self._last_emit is not None and (now - self._last_emit) < self._interval_s:
            self._dropped_interval += 1
            return
        if frame_id is not None and frame_id == self._last_frame_id:
            # 同一真实帧重复到达：推进的是网络，不是新观测 -> 不计入。
            self._dropped_repeat += 1
            return
        self._last_emit = now
        self._last_frame_id = frame_id
        self._emitted += 1
        if self._wall:
            self._win_emitted += 1
        self._session.accumulate(frame_bgr, ts_mono)

    # ------------------------------------------------------- 墙钟分桶（anchor=wall）
    def _bucket_start(self, ts_wall):
        """按 cadence_minutes 对齐的桶起点（默认 60min=整点）。"""
        period = max(int(self._cadence_s), 1)          # 秒
        start_epoch = (int(ts_wall.timestamp()) // period) * period
        return datetime.fromtimestamp(start_epoch, tz=ts_wall.tzinfo)

    def _bucket_id_of(self, start) -> str:
        if int(self._cadence_s) % 3600 == 0:
            return start.strftime("%Y-%m-%dT%H")
        return start.strftime("%Y-%m-%dT%H:%M")

    def tick(self, ts_wall) -> None:
        """墙钟推进点，由运行时循环调用（与是否有帧无关）。

        负责开窗（每桶起点）与结算（每桶的 burst 边界）。断流时本方法照常
        被调用，因此桶结果不会因为 on_frame 不再触发而丢失。
        """
        if self._released or not (self._overnight and self._wall):
            return
        start = self._bucket_start(ts_wall)
        bid = self._bucket_id_of(start)
        if self._bucket_id != bid:
            if self._window_open:  # 上一个桶未结算：按边界补结算（可能 silent）
                self._finalize_bucket(self._window_end or start)
            self._bucket_id = bid
            self._window_start = start
            self._window_end = start + timedelta(seconds=self._burst_s)
            self._window_open = True
            self._win_received = 0
            self._win_emitted = 0
            self._last_frame_id = None
            self._latest_frame = None
            self._latest_id = None          # 防止开窗首格误用上一桶的陈旧帧
            self._grid_next = start + timedelta(seconds=self._interval_s)
            self._session.begin_burst(start)
        # 固定网格采样：每个网格点取"当前最新帧"，无新帧则跳过（真实空档）。
        # 采样时间戳用连续 monotonic（跨桶不回绕），供基线 EMA 计算真实 dt。
        guard = 0
        while self._window_open and ts_wall >= self._grid_next and guard < 100000:
            guard += 1
            fid = self._latest_id
            has_new = ((fid is not None and fid != self._last_frame_id)
                       or (fid is None and self._latest_frame is not None))
            if has_new:
                self._last_frame_id = fid
                self._emitted += 1
                self._win_emitted += 1
                self._session.accumulate(self._latest_frame,
                                         self._clock.monotonic())
            else:
                self._dropped_interval += 1
            self._grid_next = self._grid_next + timedelta(seconds=self._interval_s)
        if self._window_open and ts_wall >= self._window_end:
            self._finalize_bucket(self._window_end)

    def _finalize_bucket(self, ts_wall) -> None:
        if not self._window_open:
            return
        self._window_open = False
        bid = self._bucket_id
        if bid is not None and bid == self._last_emitted_bucket:
            return  # 幂等：本桶已出图（重启后），不重复 publish
        self._bursts_done += 1
        try:
            rep = self._session.snapshot(ts_wall, force=True)
        except Exception:
            logger.exception("[night] bucket snapshot failed")
            rep = None
        if rep is None:
            rep = self._bucket_fallback(ts_wall, bid)
        scheduled_steps = max(int(round(self._burst_s / self._interval_s)), 1)
        self._night_scheduled += scheduled_steps
        self._night_emitted += self._win_emitted
        md = rep.metadata
        md[R.M_ADAPTER] = self.stats()
        md["bucket"] = bid
        md["scheduled_duration_s"] = self._burst_s
        md["scheduled_steps"] = scheduled_steps
        md["actual_frames"] = int(self._win_received)
        md["unique_frames"] = int(self._win_emitted)
        md["coverage"] = round(self._win_emitted / float(scheduled_steps), 4)
        md["silent_bucket"] = (self._win_emitted == 0)
        with self._pending_lock:
            self._pending.append(rep)
        self._last_emitted_bucket = bid
        self.save_state(ts_wall)

    def _bucket_fallback(self, ts_wall, bid):
        """无预热/无图时仍产出桶结果，避免静默丢桶。"""
        created = (ts_wall.isoformat() if hasattr(ts_wall, "isoformat")
                   else str(ts_wall))
        meta = {
            "bucket": bid,
            "scheduled_duration_s": self._burst_s,
            "actual_frames": int(self._win_received),
            "unique_frames": int(self._win_emitted),
            "coverage": 0.0,
            "silent_bucket": True,
            R.M_REASON: "no_samples",
        }
        meta[R.M_ADAPTER] = self.stats()
        return R.build_report(self._camera_id, created, None, meta,
                              status=R.STATUS_FAILED,
                              report_type=R.REPORT_TYPE_BURST)

    def _end_burst(self, ts_wall) -> None:
        """Burst boundary: snapshot cumulative heatmap, reset is session-side."""
        self._burst_active = False
        self._bursts_done += 1
        try:
            rep = self._session.snapshot(ts_wall)
        except Exception:  # never raise into the runtime loop
            logger.exception("[night] burst snapshot failed")
            rep = None
        if rep is not None:
            rep.metadata[R.M_ADAPTER] = self.stats()
            with self._pending_lock:
                self._pending.append(rep)
        self.save_state(ts_wall)

    def _maybe_periodic_save(self, now) -> None:
        """连续(非 burst)模式下的定期快照，避免只在 freeze 时才落盘。"""
        if self._store is None:
            return
        if self._last_save_mono is None:
            self._last_save_mono = now
            return
        if now - self._last_save_mono < self._save_interval_s:
            return
        self._last_save_mono = now
        self.save_state()

    def take_reports(self) -> list:
        """Drain burst reports produced since the last call (non-blocking)."""
        with self._pending_lock:
            out = self._pending
            self._pending = []
        return out

    def stats(self) -> dict:
        return {
            "overnight": self._overnight,
            "cadence_s": self._cadence_s,
            "burst_s": self._burst_s,
            "bursts_done": self._bursts_done,
            "burst_active": self._burst_active,
            "anchor": self._anchor,
            "bucket": self._bucket_id,
            "window_open": self._window_open,
            "last_emitted_bucket": self._last_emitted_bucket,
            "observations_received": self._received,
            "observations_processed": self._emitted,
            "observations_dropped": (self._dropped_interval + self._dropped_burst
                                     + self._dropped_repeat),
            "dropped_interval": self._dropped_interval,
            "dropped_burst": self._dropped_burst,
            "dropped_repeat": self._dropped_repeat,
            "restored": self._restored,
            "persisted": self._store is not None,
            "night_scheduled": int(self._night_scheduled),
            "night_emitted": int(self._night_emitted),
            "night_coverage": (round(self._night_emitted
                                     / float(self._night_scheduled), 4)
                               if self._night_scheduled else None),
        }

    def freeze(self) -> None:
        if self._released or self._frozen:
            return
        self._frozen = True
        if self._overnight and self._burst_active:
            # Freeze can cut a burst short; drop the partial burst silently
            # (the morning finalize still reports the accumulated spatial map).
            self._burst_active = False
        self._session.freeze()
        self.save_state()

    def finalize(self, base_frame_bgr, ts_wall):
        rep = self._session.finalize(base_frame_bgr, ts_wall)
        if rep is not None:
            rep.metadata[R.M_ADAPTER] = self.stats()
            if self._night_scheduled:
                rep.metadata["coverage"] = round(
                    self._night_emitted / float(self._night_scheduled), 4)
        self._archive_now(ts_wall)
        if self._store is not None and self._night_key is not None:
            self._store.clear(self._camera_id, self._night_key)
        return rep

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._session.release()
