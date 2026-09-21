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
        self._received = 0
        self._dropped_interval = 0
        self._dropped_burst = 0

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

    def on_frame(self, frame_bgr, ts_wall, ts_mono) -> None:
        if self._released or self._frozen:
            return
        now = self._clock.monotonic()
        if self._overnight:
            if self._night_start_mono is None:
                self._night_start_mono = now
            elapsed = now - self._night_start_mono
            in_burst = (elapsed % self._cadence_s) < self._burst_s
            if in_burst and not self._burst_active:
                self._burst_active = True
                self._session.begin_burst(ts_wall)
            elif not in_burst and self._burst_active:
                self._end_burst(ts_wall)
            self._received += 1
            if not in_burst:
                self._dropped_burst += 1
                return
        else:
            self._received += 1
            self._maybe_periodic_save(now)

        if self._last_emit is not None and (now - self._last_emit) < self._interval_s:
            self._dropped_interval += 1
            return
        self._last_emit = now
        self._emitted += 1
        self._session.accumulate(frame_bgr, ts_mono)

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
            "observations_received": self._received,
            "observations_processed": self._emitted,
            "observations_dropped": self._dropped_interval + self._dropped_burst,
            "dropped_interval": self._dropped_interval,
            "dropped_burst": self._dropped_burst,
            "restored": self._restored,
            "persisted": self._store is not None,
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
        self._archive_now(ts_wall)
        if self._store is not None and self._night_key is not None:
            self._store.clear(self._camera_id, self._night_key)
        return rep

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._session.release()
