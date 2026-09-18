"""NightAdapter: interval-gated feed of a NightSession (Wave 1, Agent D).

The adapter owns the ``sampling.interval_ms`` cadence using the injected
``clock`` (``schedule.Clock`` protocol); the session owns accumulation and
knows nothing about time gating.  One adapter per camera per night.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.contracts import NightLampSpec  # noqa: E402

from night_lamp.session import NightSession  # noqa: E402


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

    @property
    def emitted(self) -> int:
        return self._emitted

    def on_frame(self, frame_bgr, ts_wall, ts_mono) -> None:
        if self._released or self._frozen:
            return
        now = self._clock.monotonic()
        if self._last_emit is not None and (now - self._last_emit) < self._interval_s:
            return
        self._last_emit = now
        self._emitted += 1
        self._session.accumulate(frame_bgr, ts_mono)

    def freeze(self) -> None:
        if self._released or self._frozen:
            return
        self._frozen = True
        self._session.freeze()

    def finalize(self, base_frame_bgr, ts_wall):
        return self._session.finalize(base_frame_bgr, ts_wall)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._session.release()
