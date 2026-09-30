"""NightSession + online accumulation for the night heatmap (Wave 1, Agent D).

One instance per camera per night.  It consumes already-sampled BGR frames
(``NightAdapter`` owns the interval gate), accumulates the bounded statistics
the nightly map needs, and at 07:00 turns them into a ``Report`` by calling
*only* the pure functions of ``night_lamp.tools.nightly_map`` and
``night_lamp.periodicity``.  Neither module is modified.

Design (and the online-vs-golden deltas this file makes explicit):

* ``warmup`` collects the first ``baseline.frames`` sampled frames, builds the
  per-pixel baseline ``base`` and the render background ``bg``, then drops the
  warm cache immediately (never grows with night length).
* ``on`` / ``peak`` are full-frame float32 accumulators updated per sampled
  frame, mirroring ``nightly_map.duty_swing``.  ``peak`` *is* the swing map
  used for reflection/host search at finalize.
* ``CandidateDiscovery`` (online, rolling) and ``TemporalSeries`` (storage)
  are deliberately separate objects.  Discovery watches the rolling duty and
  decides which pixels get a time series; the series only stores.  At finalize
  the authoritative candidate components still come from
  ``nightly_map.split_comps(final duty)`` -- the online set is *never* used to
  prove equivalence, only to decide what to track.
* ``max_candidates`` overflow sets ``candidate_overflow`` in metadata (never
  silent) but does *not* gate ``Report.status``: the rendered heatmap comes
  from the full-frame duty accumulator, so the number of light spots in the
  scene is not a quality verdict on the output.
* ``series_cap is None`` keeps the full sequence (production default).
  A non-None cap keeps only the tail window and is marked experimental.
* Overnight burst mode (``spec.overnight.enabled``): ``begin_burst`` resets the
  temporal state (``TemporalSeries`` + ``CandidateDiscovery``) but keeps the
  spatial accumulators, so a night is a series of bounded bursts whose duty/peak
  maps merge into one cumulative heatmap.  ``snapshot`` emits that heatmap
  without freezing or releasing.  Disabled -> one continuous session.
* ``release()`` drops every accumulator so a failed/hung finalize cannot leak
  into the next night.

Known online-vs-golden differences (reported in metadata, not hidden):
  - warm baseline = first N samples, golden = N spread over the whole night;
  - series start at discovery time, golden starts at sample 0;
  - lag windows use the design grid rate (1/interval), golden uses video fps/step;
  - reflection hosts are matched to the nearest tracked pixel;
  - dynamic-blob lamp cores are only tracked if discovered while rolling.
"""
from __future__ import annotations

import functools
import os
import sys
import threading

import cv2 as cv
import numpy as np
from loguru import logger

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.contracts import AlignmentStatus, NightLampSpec  # noqa: E402

from night_lamp import report as R  # noqa: E402
from night_lamp.periodicity import ac_limited, clean_train, onsets_of  # noqa: E402
from night_lamp.persist import (  # noqa: E402
    EVIDENCE_VERSION as _EVIDENCE_VERSION,
    STATE_VERSION as _STATE_VERSION,
)
from night_lamp.tools import nightly_map as nm  # noqa: E402


class _NightState:
    """Inlined from the production ``night_lamp.night_state`` (pure hysteresis).

    The deploy world must not import ``night_lamp.night_state`` (boundary scan
    forbids it) and only this session consumes it, so the 40-line state machine
    is kept here verbatim to stay self-contained.
    """

    NIGHT_VERSION = "p1-provisional-20260908"
    TWILIGHT = "TWILIGHT"
    NIGHT = "NIGHT"

    @staticmethod
    def update(state, count, g, t_enter, t_exit, persistence):
        entered = False
        if state == _NightState.TWILIGHT:
            count = count + 1 if g < t_enter else 0
            if count >= persistence:
                state, entered = _NightState.NIGHT, True
        else:
            if g > t_exit:
                state, count = _NightState.TWILIGHT, 0
        return state, count, entered


NS = _NightState

_M_NIGHT_STATE = "night_state"
_M_NIGHT_QUALIFIED = "night_qualified"
_M_NIGHT_GATE = "night_gate"

_SERIES_INIT_COLS = 256
_DISCOVERY_EVERY = 50          # samples between rolling discovery passes
_MIN_DISCOVERY_SAMPLES = 250   # ~40s before the first pass (kill early speckle)
_WATCH_MIN_ON = 3              # rolling ON floor for a tracked core
_DEDUP_R = 4                   # px; one tracked core per lamp, not per pass
_BASE_BLOCK_ROWS = 256         # warm baseline is built blockwise to bound RAM
_CHANGE_POINT_DELTA = 20.0     # global-median jump (gray) => reset EMA baseline


def _synchronized(method):
    """Serialize a NightSession public method on its own RLock.

    ``tick`` runs on the camera worker thread while ``freeze``/``finalize``/
    ``save_state`` run on the runtime pump thread; the spatial accumulators and
    the per-hour flash evidence are shared, so they must not interleave.
    """
    @functools.wraps(method)
    def _wrap(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return _wrap


class TemporalSeries:
    """Bounded per-pixel corrected-gray time series.

    Rows are tracked pixels; columns are accepted samples.  ``cap=None`` keeps
    the full sequence (grows by doubling).  A cap keeps only the last ``cap``
    samples (experimental tail window).  Each row records the sample index at
    which it was first tracked so finalize can align rows to a common start
    instead of zero-padding.
    """

    _CELL = _DEDUP_R

    def __init__(self, max_points, cap=None):
        self.max_points = max(1, int(max_points))
        self.cap = None if cap is None else max(1, int(cap))
        self._pys = np.full(self.max_points, -1, np.int32)
        self._pxs = np.full(self.max_points, -1, np.int32)
        self._starts = np.zeros(self.max_points, np.int64)
        self._grid = {}
        self._n_points = 0
        self._n_total = 0
        if self.cap is None:
            self._buf = np.zeros((self.max_points, _SERIES_INIT_COLS), np.int16)
        else:
            self._buf = np.zeros((self.max_points, self.cap), np.int16)
            self._write = 0

    @property
    def n_points(self):
        return self._n_points

    @property
    def full(self):
        return self._n_points >= self.max_points

    def has_room(self, n=1):
        """True iff ``n`` more rows fit; a False answer is a capacity drop.

        ``track`` can only add one row at a time and returns ``None`` when the
        series is full, so callers that claim several points per candidate must
        ask *before* tracking anything.  This is what makes the overflow flag
        deterministic instead of depending on how many of N tracks happened to
        land.
        """
        return (self._n_points + int(n)) <= self.max_points

    def has_data(self, r):
        return r is not None and int(self._starts[r]) < self._n_total

    def track(self, y, x, kind=None):
        if self._n_points >= self.max_points:
            return None
        r = self._n_points
        self._pys[r], self._pxs[r] = int(y), int(x)
        self._starts[r] = self._n_total
        self._grid.setdefault((int(y) // self._CELL, int(x) // self._CELL),
                              []).append(r)
        self._n_points += 1
        return r

    def lookup(self, y, x, radius=_DEDUP_R):
        rad = int(radius)
        rad2 = rad * rad
        cy, cx = int(y) // self._CELL, int(x) // self._CELL
        reach = (rad + self._CELL - 1) // self._CELL
        best, best_d2 = None, rad2 + 1
        for gy in range(cy - reach, cy + reach + 1):
            for gx in range(cx - reach, cx + reach + 1):
                for i in self._grid.get((gy, gx), ()):
                    dy = int(self._pys[i]) - int(y)
                    dx = int(self._pxs[i]) - int(x)
                    d2 = dy * dy + dx * dx
                    if d2 <= rad2 and d2 < best_d2:
                        best, best_d2 = i, d2
        return best

    def append(self, gray, mask, off):
        npts = self._n_points
        if npts:
            ys, xs = self._pys[:npts], self._pxs[:npts]
            vals = gray[ys, xs].astype(np.int16)
            vals -= np.where(mask[ys, xs], 0, off).astype(np.int16)
            if self.cap is None:
                if self._n_total >= self._buf.shape[1]:
                    self._grow()
                self._buf[:npts, self._n_total] = vals
            else:
                self._buf[:npts, self._write] = vals
                self._write = (self._write + 1) % self.cap
        self._n_total += 1

    def _grow(self):
        cols = self._buf.shape[1]
        grown = np.zeros((self.max_points, cols * 2), np.int16)
        grown[:, :cols] = self._buf
        self._buf = grown

    def series(self, r):
        """This row's own tracked values from its first sample (no alignment)."""
        if r is None:
            return np.zeros(0, np.int16)
        return self._tail(r)[0]

    def _tail(self, r):
        if self.cap is None:
            return self._buf[r, self._starts[r]:self._n_total], int(self._starts[r])
        retained = min(self._n_total, self.cap)
        first = self._n_total - retained
        if self._n_total < self.cap:
            vals = self._buf[r, :self._n_total].copy()
        else:
            vals = np.concatenate([self._buf[r, self._write:],
                                   self._buf[r, :self._write]])
        return vals, first

    def aligned_matrix(self, row_ids):
        """Rows aligned to their common latest start; missing rows stay zero."""
        ids = [r for r in row_ids if r is not None]
        if not ids or self._n_total <= 0:
            return np.zeros((len(row_ids), 0), np.int16)
        tails = {}
        starts = []
        for r in ids:
            vals, first = self._tail(r)
            tails[r] = (vals, first)
            if vals.size:
                starts.append(first)
        if not starts:
            return np.zeros((len(row_ids), 0), np.int16)
        start_common = max(starts)
        t = self._n_total - start_common
        if t <= 0:
            return np.zeros((len(row_ids), 0), np.int16)
        out = np.zeros((len(row_ids), t), np.int16)
        for i, r in enumerate(row_ids):
            if r is None:
                continue
            vals, first = tails[r]
            if vals.size == 0:
                continue
            off = start_common - first
            out[i] = vals[off:off + t]
        return out

    def memory_bytes(self):
        if self._buf is None:
            return 0
        return int(self._buf.nbytes + self._pys.nbytes + self._pxs.nbytes
                   + self._starts.nbytes)

    def release(self):
        self._buf = None
        self._pys = None
        self._pxs = None
        self._starts = None
        self._grid.clear()
        self._n_points = 0


class CandidateDiscovery:
    """Online rolling candidate watch, separate from series storage.

    Emits the pixels that deserve a time series: small flashing components
    (candidates) and dynamic-blob lamp cores.  Capacity is bounded by
    ``max_candidates``; anything beyond it sets ``overflow`` and is counted,
    never silently truncated.
    """

    def __init__(self, series, max_candidates):
        self._series = series
        self._max_candidates = int(max_candidates)
        self._cand_cells = {}
        self._dyn_cells = {}
        self._dedup_r = _DEDUP_R
        self.overflow = False
        self.n_dropped = 0
        self.n_cand_dropped = 0
        self.n_dyn_dropped = 0
        self.n_passes = 0
        self.first_discovery_sample = None
        self.n_components_seen = 0
        self.n_candidate_cores = 0
        self.n_dyn_cores = 0

    def _dup(self, cells, y, x):
        r = self._dedup_r
        rr = r * r
        cy, cx = y // r, x // r
        for gy in range(cy - 1, cy + 2):
            for gx in range(cx - 1, cx + 2):
                for py, px in cells.get((gy, gx), ()):
                    dy, dx = py - y, px - x
                    if dy * dy + dx * dx <= rr:
                        return True
        return False

    def _claim(self, cells, y, x):
        if self._dup(cells, y, x):
            return False
        cells.setdefault((y // self._dedup_r, x // self._dedup_r),
                         []).append((y, x))
        return True

    def _overflow(self, kind="candidate"):
        self.overflow = True
        self.n_dropped += 1
        if kind == "dyn":
            self.n_dyn_dropped += 1
        else:
            self.n_cand_dropped += 1

    def observe(self, duty, on, n_seen, swing):
        if n_seen < _MIN_DISCOVERY_SAMPLES:
            return
        self.n_passes += 1
        if self.first_discovery_sample is None:
            self.first_discovery_sample = int(n_seen)
        dyn, lab, stats, cand = nm.split_comps(duty)
        cand = sorted(cand, key=lambda c: -stats[c, cv.CC_STAT_AREA])
        self.n_components_seen += len(cand)
        groups = nm.label_slices(lab)
        for c in cand:
            ys, xs = groups.get(int(c), (None, None))
            if ys is None or ys.size == 0:
                continue
            jd = int(np.argmax(duty[ys, xs]))
            y, x = int(ys[jd]), int(xs[jd])
            if int(on[y, x]) < _WATCH_MIN_ON or self._dup(self._cand_cells, y, x):
                continue
            if (self.n_candidate_cores >= self._max_candidates
                    or not self._series.has_room(3)):
                self._overflow("candidate")
                continue
            js = int(np.argmax(swing[ys, xs]))
            sy, sx = int(ys[js]), int(xs[js])
            host = nm.host_of(swing, sy, sx, nm.NEIGH_R, nm.SELF_R) or (sy, sx)
            rows = (self._series.track(sy, sx),
                    self._series.track(y, x),
                    self._series.track(host[0], host[1]))
            if any(r is None for r in rows):
                self._overflow("candidate")
                continue
            self._claim(self._cand_cells, y, x)
            self.n_candidate_cores += 1
        for (y, x) in nm.dyn_samples(duty, dyn):
            if not self._series.has_room(1):
                self._overflow("dyn")
                continue
            if self._claim(self._dyn_cells, y, x):
                if self._series.track(y, x) is None:
                    self._overflow("dyn")
                    continue
                self.n_dyn_cores += 1

    def release(self):
        self._cand_cells.clear()
        self._dyn_cells.clear()


class NightSession:
    """Bounded, per-camera, per-night accumulator + finalize."""

    def __init__(self, camera_id: str, spec: NightLampSpec):
        self._camera_id = camera_id
        self._spec = spec
        self._lock = threading.RLock()
        self._max_candidates = int(spec.memory.max_candidates)
        self._series_cap = spec.memory.series_cap
        self._max_points = 3 * self._max_candidates + nm.DYN_MAX_PEAK
        self._series = TemporalSeries(self._max_points, cap=self._series_cap)
        self._discovery = CandidateDiscovery(self._series, self._max_candidates)
        self._baseline_frames = max(1, int(spec.baseline.frames))
        self._warmup_s = float(spec.baseline.warmup_s)
        self._ema_half_life_s = float(
            getattr(spec.baseline, "ema_half_life_s", 1800.0) or 0.0)
        _interval_s = max(int(spec.sampling.interval_ms), 1) / 1000.0
        self._ema_alpha = (1.0 - 0.5 ** (_interval_s / self._ema_half_life_s)
                           if self._ema_half_life_s > 0 else 1.0)
        self._last_med = None
        self._base_reset = False
        self._period = spec.periodicity
        self._alignment = spec.alignment
        self._discovery_every = _DISCOVERY_EVERY

        # Overnight burst mode: spatial accumulators persist, temporal state is
        # reset per burst by ``begin_burst``.  Duck-typed so tests constructing
        # a bare NightLampSpec keep the old single-session behaviour.
        ov = getattr(spec, "overnight", None)
        self._overnight = bool(getattr(ov, "enabled", False))
        self._cadence_minutes = int(getattr(ov, "cadence_minutes", 60))
        self._burst_seconds = int(getattr(ov, "burst_seconds", 120))
        self._burst_first_ts = None
        self._burst_last_ts = None
        self._burst_n = 0
        self._burst_started_wall = None
        self._burst_ended_wall = None
        self._overflow_any = False
        self._last_snapshot_n = 0

        self._mask = None
        self._h = self._w = 0
        self._warm_stack = None
        self._warm_n = 0
        self._warm_meds = []
        self._first_warm_ts = None
        self._warm_done = False
        self._detrend = None
        self._base = None
        self._bg = None
        self._on = None
        self._peak = None
        self._excess = None
        self._on_hit = None
        self._flash_union = None
        self._flash_count = None
        self._n_flash_buckets = 0

        self._n_seen = 0
        self._first_ts = None
        self._last_ts = None
        self._frozen = False
        self._released = False
        self._restored = False
        self._dropped_after_freeze = 0
        self._init_gate(spec.night_gate)
        # 水马灯带 ROI（空 = 关闭，旧行为）：只在水马行附近找警示灯。
        self._lamp_roi = dict(getattr(spec, "lamp_roi", None) or {})
        self._lamp_roi_frac = None

    def _merge_lamp_roi(self):
        """灯带 ROI 并入排除掩膜；失败则 fail-open（记错，不杀死本夜）。"""
        if not self._lamp_roi or self._mask is None:
            return
        try:
            roi = nm.lamp_roi_mask(
                self._h, self._w, self._lamp_roi["calibration"],
                int(self._lamp_roi.get("dilate_px", 30)),
                int(self._lamp_roi.get("up_px", 40)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("lamp_roi 构建失败，本夜按全帧跑: {}", exc)
            self._lamp_roi = {}
            return
        self._mask = self._mask | ~roi
        self._lamp_roi_frac = round(float(roi.mean()), 4)

    def _init_gate(self, gate):
        """Online night-qualification state (O(1); no full-night median list)."""
        gate = gate or {}
        self._gate_enter = None
        self._gate_exit = None
        self._gate_persistence = 1
        self._gate_configured = False
        self._gate_valid = False
        try:
            te, tx = gate.get("enter_threshold"), gate.get("exit_threshold")
            if te is not None and tx is not None:
                self._gate_enter = float(te)
                self._gate_exit = float(tx)
                self._gate_persistence = max(1, int(gate.get("persistence", 1)))
                self._gate_configured = True
                self._gate_valid = self._gate_enter < self._gate_exit
        except (TypeError, ValueError):
            self._gate_configured = self._gate_valid = False
        self._gate_state = NS.TWILIGHT
        self._gate_count = 0
        self._gate_transitions = 0
        self._gate_samples = 0

    @property
    def sample_count(self) -> int:
        return self._n_seen

    @property
    def candidate_overflow(self) -> bool:
        return self._overflow_any or self._discovery.overflow

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def restored(self) -> bool:
        return self._restored

    @_synchronized
    def dump_state(self):
        """序列化跨重启所需的累计状态；尚未完成预热时返回 None。"""
        if not self._warm_done or self._base is None:
            return None
        return {
            "version": _STATE_VERSION,
            "base": self._base,
            "bg": self._bg,
            "on": self._on,
            "peak": self._peak,
            "detrend": np.float32(0.0 if self._detrend is None else self._detrend),
            "n_seen": np.int64(self._n_seen),
            "overflow_any": np.bool_(self._overflow_any),
            "gate_state": self._gate_state,
            "gate_count": np.int64(self._gate_count),
            "gate_samples": np.int64(self._gate_samples),
            "gate_transitions": np.int64(self._gate_transitions),
            "gate_configured": np.bool_(self._gate_configured),
            "gate_valid": np.bool_(self._gate_valid),
            "gate_enter": np.float32(self._nan(self._gate_enter)),
            "gate_exit": np.float32(self._nan(self._gate_exit)),
            "gate_persistence": np.int64(self._gate_persistence),
            "flash_union": (self._flash_union if self._flash_union is not None
                            else np.zeros((self._h, self._w), np.uint8)),
            "flash_count": (self._flash_count if self._flash_count is not None
                            else np.zeros((self._h, self._w), np.uint8)),
            "n_flash_buckets": np.int64(self._n_flash_buckets),
            "evidence_version": np.int64(_EVIDENCE_VERSION),
        }

    @_synchronized
    def load_state(self, state) -> bool:
        """从 dump_state 的快照恢复累计，跳过预热；时间态仍由本进程重建。"""
        if not state or self._warm_done:
            return False
        try:
            base = np.asarray(state["base"], np.float32)
            bg = np.asarray(state["bg"], np.float32)
            on = np.asarray(state["on"], np.float32)
            peak = np.asarray(state["peak"], np.float32)
        except Exception:
            logger.exception("night load_state 解析失败")
            return False
        if base.ndim != 2 or base.size == 0:
            return False
        if not (bg.shape == base.shape == on.shape == peak.shape):
            logger.warning("night load_state 形状不一致，忽略")
            return False

        self._h, self._w = int(base.shape[0]), int(base.shape[1])
        self._mask = nm.osd_mask(self._h, self._w)
        self._merge_lamp_roi()
        self._base, self._bg, self._on, self._peak = base, bg, on, peak
        self._detrend = float(state["detrend"])
        self._n_seen = int(state["n_seen"])
        self._warm_done = True
        self._warm_stack = None
        self._warm_meds = None
        self._overflow_any = bool(state["overflow_any"])
        self._restore_gate(state)
        self._restore_evidence(state, base.shape)
        self._restored = True
        return True

    def _restore_gate(self, state):
        self._gate_state = str(state["gate_state"])
        self._gate_count = int(state["gate_count"])
        self._gate_samples = int(state["gate_samples"])
        self._gate_transitions = int(state["gate_transitions"])
        self._gate_configured = bool(state["gate_configured"])
        self._gate_valid = bool(state["gate_valid"])
        enter, exit_ = float(state["gate_enter"]), float(state["gate_exit"])
        self._gate_enter = None if enter != enter else enter
        self._gate_exit = None if exit_ != exit_ else exit_
        self._gate_persistence = int(state["gate_persistence"])

    def _restore_evidence(self, state, shape):
        """Restore per-hour flash evidence when version- and shape-matched.

        ``n_flash_buckets`` is the sentinel for "evidence exists": a zero-count
        state (e.g. continuous mode, where no BURST snapshot ever accumulates)
        must leave ``_flash_union`` as None so the morning finalize keeps its
        per-pixel fallback instead of rendering an empty union.
        """
        self._n_flash_buckets = 0
        if int(state.get("evidence_version", 0)) != _EVIDENCE_VERSION:
            return
        n = int(state.get("n_flash_buckets", 0))
        fu = state.get("flash_union")
        if n <= 0 or fu is None or np.asarray(fu).shape != shape:
            return
        self._flash_union = np.asarray(fu, np.uint8)
        self._n_flash_buckets = n
        fc = state.get("flash_count")
        if fc is not None and np.asarray(fc).shape == shape:
            self._flash_count = np.asarray(fc, np.uint8)

    @staticmethod
    def _nan(value):
        return np.nan if value is None else value

    @_synchronized
    def accumulate(self, frame_bgr, ts_mono: float) -> None:
        if self._released:
            raise RuntimeError("NightSession is released")
        if self._frozen:
            self._dropped_after_freeze += 1
            return
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if self._mask is None:
            self._h, self._w = gray.shape
            self._mask = nm.osd_mask(self._h, self._w)
            self._merge_lamp_roi()
        prev_ts = self._last_ts
        if self._first_ts is None:
            self._first_ts = float(ts_mono)
        self._last_ts = float(ts_mono)
        if self._burst_first_ts is None:
            self._burst_first_ts = float(ts_mono)
        self._burst_last_ts = float(ts_mono)
        self._burst_n += 1
        med = float(np.median(gray))
        self._n_seen += 1
        self._update_gate(med)

        if not self._warm_done:
            if self._first_warm_ts is None:
                self._first_warm_ts = float(ts_mono)
            if self._warm_stack is None:
                self._warm_stack = np.empty(
                    (self._baseline_frames, self._h, self._w), np.uint8)
            if self._warm_n < self._baseline_frames:
                self._warm_stack[self._warm_n] = gray
                self._warm_n += 1
            self._warm_meds.append(med)
            if self._warm_ready(ts_mono):
                self._finish_warmup()
            return

        dt = None if prev_ts is None else max(float(ts_mono) - prev_ts, 0.0)
        self._process(gray, med, dt)
        if self._n_seen % self._discovery_every == 0:
            self._run_discovery()

    def _warm_ready(self, ts_mono):
        if self._warm_n >= self._baseline_frames:
            return True
        return (self._first_warm_ts is not None
                and (ts_mono - self._first_warm_ts) >= self._warmup_s
                and self._warm_n > 0)

    def _finish_warmup(self):
        n = self._warm_n
        if n <= 0 or self._warm_stack is None:
            self._warm_done = True
            self._warm_stack = None
            self._warm_meds = None
            return
        warm_meds = self._warm_meds
        self._detrend = float(np.median(warm_meds))
        offs = [int(round(m - self._detrend)) for m in warm_meds]
        fh, fw = self._h, self._w
        base = np.empty((fh, fw), np.float32)
        for y0 in range(0, fh, _BASE_BLOCK_ROWS):
            y1 = min(fh, y0 + _BASE_BLOCK_ROWS)
            sub = self._warm_stack[:n, y0:y1, :].astype(np.int16)
            for i in range(n):
                sub[i] -= offs[i]
            base[y0:y1] = np.median(sub, axis=0, overwrite_input=True)
            del sub
        self._base = base
        self._on = np.zeros_like(base, np.float32)
        self._peak = np.zeros_like(base, np.float32)
        for i in range(n):
            self._process(self._warm_stack[i], warm_meds[i])
        self._bg = np.median(self._warm_stack[:n], axis=0,
                             overwrite_input=True).astype(np.float32)
        self._warm_stack = None
        self._warm_meds = None
        self._warm_done = True

    def _tracked_row(self, y, x):
        r = self._series.lookup(int(y), int(x))
        return r if self._series.has_data(r) else None

    def _process(self, gray, med, dt=None):
        if (self._last_med is not None
                and abs(med - self._last_med) > _CHANGE_POINT_DELTA):
            self._base_reset = True
        self._last_med = float(med)
        off = int(round(med - self._detrend))
        if self._excess is None:
            self._excess = np.empty(gray.shape, np.float32)
        ex = self._excess
        ex[:] = gray
        ex[self._mask] = 0.0
        ex -= off
        ex -= self._base
        if self._on_hit is None:
            self._on_hit = np.empty(gray.shape, bool)
        np.greater_equal(ex, nm.ON_DELTA, out=self._on_hit)
        np.add(self._on, self._on_hit, out=self._on)
        np.maximum(self._peak, ex, out=self._peak)
        self._update_base(gray, off, dt)
        self._series.append(gray, self._mask, off)

    def _update_base(self, gray, off, dt):
        """Slow per-pixel EMA baseline, foreground-protected.

        Only pixels that are not currently ON are pulled toward the frame, so a
        lamp's own flashes are never learned into the background.  ``dt`` is the
        wall seconds since the previous sample (None during warmup replay);
        a large global-median jump resets the whole pattern to the current frame
        (IR / exposure switch).
        """
        if self._base is None:
            return
        det = gray.astype(np.float32)
        det[self._mask] = 0.0
        det -= off
        if self._base_reset:
            self._base[~self._mask] = det[~self._mask]
            self._base_reset = False
            return
        if dt is None or dt <= 0 or self._ema_half_life_s <= 0:
            alpha = self._ema_alpha
        else:
            alpha = 1.0 - 0.5 ** (dt / self._ema_half_life_s)
        upd = ~self._on_hit & ~self._mask
        self._base[upd] = alpha * det[upd] + (1.0 - alpha) * self._base[upd]

    def _run_discovery(self):
        if not self._warm_done or self._n_seen <= 0 or self._on is None:
            return
        duty = self._on / float(self._n_seen)
        self._discovery.observe(duty, self._on, self._n_seen, self._peak)
        self._overflow_any = self._overflow_any or self._discovery.overflow

    def _accumulate_flash(self, flash):
        """OR one burst's flash mask into the night evidence.

        Frozen semantics: the morning map is the OR of per-hour presence, not
        a sustained-duration map; ``_flash_count`` keeps the per-pixel hit
        count for diagnostics only.
        """
        if self._flash_union is None:
            self._flash_union = np.zeros(flash.shape, np.uint8)
            self._flash_count = np.zeros(flash.shape, np.uint8)
        self._flash_union |= flash.astype(np.uint8)
        self._flash_count[flash] += 1
        self._n_flash_buckets += 1

    def _update_gate(self, med):
        """Stream the night_gate hysteresis; O(1) state, no median history."""
        self._gate_samples += 1
        if not (self._gate_configured and self._gate_valid):
            return
        prev = self._gate_state
        self._gate_state, self._gate_count, _entered = NS.update(
            self._gate_state, self._gate_count, float(med), self._gate_enter,
            self._gate_exit, self._gate_persistence)
        if self._gate_state != prev:
            self._gate_transitions += 1

    @_synchronized
    def begin_burst(self, ts_wall=None) -> None:
        """Start a burst: keep spatial accumulators, reset temporal tracking.

        Called by the overnight adapter at each burst boundary (and once at
        night start).  ``_base/_bg/_on/_peak/_mask/_n_seen`` persist so the
        heatmap keeps accumulating across the whole night.
        """
        if self._released:
            raise RuntimeError("NightSession is released")
        self._series.release()
        self._series = TemporalSeries(self._max_points, cap=self._series_cap)
        self._discovery.release()
        self._discovery = CandidateDiscovery(self._series, self._max_candidates)
        self._burst_first_ts = None
        self._burst_last_ts = None
        self._burst_n = 0
        self._burst_started_wall = ts_wall
        self._burst_ended_wall = None

    @_synchronized
    def snapshot(self, ts_wall, force: bool = False) -> "R.Report | None":
        """Burst-end cumulative heatmap; does not freeze or release.

        Returns ``None`` when the burst added no observations since the last
        snapshot, so a silent hour does not emit a duplicate report.  ``force``
        bypasses that guard so a wall-clock bucket can still emit a (possibly
        silent) bucket report; it never fabricates observations.
        """
        if self._released:
            raise RuntimeError("NightSession is released")
        if not force and self._n_seen <= self._last_snapshot_n:
            return None
        if not self._warm_done and self._n_seen > 0:
            self._finish_warmup()
        created_at = (ts_wall.isoformat() if hasattr(ts_wall, "isoformat")
                      else str(ts_wall))
        night = self._night_qualification()
        self._burst_ended_wall = ts_wall
        self._last_snapshot_n = self._n_seen
        if self._base is None:
            return None
        try:
            return self._finalize_inner(None, created_at, night,
                                        report_type=R.REPORT_TYPE_BURST)
        except Exception as exc:  # never raise into the runtime loop
            logger.exception("night burst snapshot failed")
            meta = {
                R.M_ALIGNMENT: AlignmentStatus.REJECTED.value,
                R.M_CANDIDATE_OVERFLOW: bool(self.candidate_overflow),
                R.M_SAMPLING: self._sampling_metadata(),
                R.M_REASON: "%s: %s" % (type(exc).__name__, exc),
            }
            meta.update(self._night_meta(night))
            return R.build_report(self._camera_id, created_at, None, meta,
                                  status=R.STATUS_FAILED,
                                  report_type=R.REPORT_TYPE_BURST)

    @_synchronized
    def freeze(self) -> None:
        if self._released:
            raise RuntimeError("NightSession is released")
        if self._frozen:
            return
        if not self._warm_done and self._n_seen > 0:
            self._finish_warmup()
        self._frozen = True
        self._run_discovery()

    def _design_rate_hz(self):
        """Grid design rate: the cadence the sampler schedules.

        The autocorrelation window is a property of the *design* time base, not
        of how many samples arrived.  In wall mode the grid timestamps are
        uniform (``interval_s`` apart), so accepted-count / span understates the
        rate by the coverage factor and must not size the lag window.
        """
        return 1000.0 / max(int(self._spec.sampling.interval_ms), 1)

    def _rate_hz(self):
        """Measured effective rate (diagnostic only; never sizes the lag window)."""
        if self._overnight:
            first, last, n = self._burst_first_ts, self._burst_last_ts, self._burst_n
        else:
            first, last, n = self._first_ts, self._last_ts, self._n_seen
        if first is None or last is None:
            return 0.0
        span = last - first
        if span <= 0:
            return 0.0
        return (n - 1) / span

    def _sampling_metadata(self):
        span = (0.0 if self._first_ts is None or self._last_ts is None
                else self._last_ts - self._first_ts)
        burst_span = (0.0 if self._burst_first_ts is None
                      or self._burst_last_ts is None
                      else self._burst_last_ts - self._burst_first_ts)
        return {
            "interval_ms": int(self._spec.sampling.interval_ms),
            "design_rate_hz": round(self._design_rate_hz(), 4),
            "effective_rate_hz": round(self._rate_hz(), 4),
            "actual_samples": int(self._n_seen),
            "actual_span_s": round(span, 3),
            "mode": "tail" if self._series_cap is not None else "full",
            "series_cap": self._series_cap,
            "experimental_tail_window": self._series_cap is not None,
            "burst_samples": int(self._burst_n),
            "burst_span_s": round(burst_span, 3),
            "overnight": {
                "enabled": self._overnight,
                "cadence_minutes": self._cadence_minutes,
                "burst_seconds": self._burst_seconds,
            },
        }

    def _memory_estimate_mb(self):
        total = 0
        if self._series is not None:
            total += self._series.memory_bytes()
        for a in (self._base, self._bg, self._on, self._peak, self._mask):
            if a is not None:
                total += int(a.nbytes)
        return round(total / 1048576.0, 1)

    def _night_qualification(self):
        """Report the online night_gate hysteresis labels.

        The state is streamed in ``_update_gate`` using the exact hysteresis
        rule of ``night_state.update`` (enter_threshold < exit_threshold,
        ``persistence`` consecutive dark samples to enter, one bright sample
        above exit to leave).  No full-night median list is retained, and it
        never feeds back into the nightly map maths -- it only labels the night.
        """
        info = {
            "configured": self._gate_configured,
            "valid": self._gate_valid,
            "enter_threshold": self._gate_enter,
            "exit_threshold": self._gate_exit,
            "persistence": self._gate_persistence,
            "samples": int(self._gate_samples),
            "transitions": int(self._gate_transitions),
        }
        if not self._gate_configured:
            info["state"] = "not_configured"
            info["qualified"] = None
            return info
        if not self._gate_valid:
            info["state"] = "invalid_gate"
            info["qualified"] = None
            info["reason"] = "requires enter_threshold < exit_threshold"
            return info
        info["state"] = self._gate_state
        info["qualified"] = (self._gate_state == NS.NIGHT)
        return info

    @staticmethod
    def _night_meta(night):
        return {
            _M_NIGHT_STATE: night["state"],
            _M_NIGHT_QUALIFIED: night["qualified"],
            _M_NIGHT_GATE: {k: night[k] for k in
                            ("configured", "valid", "enter_threshold",
                             "exit_threshold", "persistence", "samples",
                             "transitions") if k in night},
        }

    @_synchronized
    def finalize(self, base_frame_bgr, ts_wall) -> R.Report:
        if self._released:
            raise RuntimeError("NightSession is released")
        if not self._warm_done and self._n_seen > 0:
            self._finish_warmup()
        created_at = (ts_wall.isoformat() if hasattr(ts_wall, "isoformat")
                      else str(ts_wall))
        night = self._night_qualification()
        if self._n_seen <= 0 or self._base is None:
            meta = {
                R.M_ALIGNMENT: AlignmentStatus.BASE_UNAVAILABLE.value,
                R.M_CANDIDATE_OVERFLOW: bool(self.candidate_overflow),
                R.M_SAMPLING: self._sampling_metadata(),
                R.M_REASON: "no_samples",
            }
            meta.update(self._night_meta(night))
            return R.build_report(self._camera_id, created_at, None, meta,
                                  status=R.STATUS_FAILED)
        try:
            return self._finalize_inner(base_frame_bgr, created_at, night)
        except Exception as exc:  # never raise into the runtime loop
            logger.exception("night finalize failed")
            meta = {
                R.M_ALIGNMENT: AlignmentStatus.REJECTED.value,
                R.M_CANDIDATE_OVERFLOW: bool(self.candidate_overflow),
                R.M_SAMPLING: self._sampling_metadata(),
                R.M_REASON: "%s: %s" % (type(exc).__name__, exc),
            }
            meta.update(self._night_meta(night))
            return R.build_report(self._camera_id, created_at, None, meta,
                                  status=R.STATUS_FAILED)

    def _finalize_inner(self, base_frame_bgr, created_at, night,
                        report_type=R.REPORT_TYPE):
        duty = self._on / float(self._n_seen)
        swing = self._peak
        dyn, lab, stats, cand_all = nm.split_comps(duty)

        rest = duty[(duty >= nm.CAND_DUTY_LO) & ~dyn]
        if rest.size == 0:
            rest = duty[duty > 0]
        top = float(np.percentile(rest, nm.PCTL)) if rest.size else 1.0
        top = top if top > 0 else 1.0

        cand_all = sorted(cand_all, key=lambda c: -stats[c, cv.CC_STAT_AREA])
        n_cand_total = len(cand_all)
        overflow = (self._overflow_any or self._discovery.overflow
                    or n_cand_total > self._max_candidates)
        cand = cand_all[:self._max_candidates]
        n_cand_dropped_final = max(0, n_cand_total - len(cand))

        groups = nm.label_slices(lab)
        cand_px = [groups.get(int(c), (np.zeros(0, np.intp), np.zeros(0, np.intp)))
                   for c in cand]
        pts, hosts, fpts = [], [], []
        for ys, xs in cand_px:
            j = int(np.argmax(swing[ys, xs]))
            oy, ox = int(ys[j]), int(xs[j])
            host = nm.host_of(swing, oy, ox, nm.NEIGH_R, nm.SELF_R)
            pts.append((oy, ox))
            hosts.append(host or (oy, ox))
            jf = int(np.argmax(duty[ys, xs]))
            fpts.append((int(ys[jf]), int(xs[jf])))

        rate = self._design_rate_hz()
        lag_lo = max(1, int(round(self._period.lag_lo_s * rate)))
        lag_hi = max(lag_lo + 1, int(round(self._period.lag_hi_s * rate)))

        rings = []
        n_flash = 0
        use_union = (report_type == R.REPORT_TYPE
                     and self._flash_union is not None
                     and self._n_flash_buckets > 0)

        if use_union:
            # Morning report: render from the accumulated per-hour evidence.
            # The temporal series is per-burst and empty after the final
            # begin_burst, so it must not be re-run here.  dim_map is skipped
            # by the Phase-0 decision: dynamic/reflection suppression has no
            # defined cross-bucket semantics, so the union mask is drawn as-is.
            flash = self._flash_union > 0
            dim = np.ones_like(duty, np.float32)
            rel = None
            n_ref = 0
            n_reflect_untracked = 0
            n_period_untracked = 0
            n_comp, flash_lab = cv.connectedComponents(flash.astype(np.uint8), 8)
            for c in range(1, n_comp):
                ys, xs = np.nonzero(flash_lab == c)
                if ys.size == 0:
                    continue
                r = max(int(ys.max() - ys.min()),
                        int(xs.max() - xs.min())) // 2 + nm.RING_PAD
                rings.append((int(xs.mean()), int(ys.mean()), r))
                n_flash += 1
        else:
            ref_rows = ([self._tracked_row(y, x) for y, x in pts]
                        + [self._tracked_row(y, x) for y, x in hosts])
            ref_series = self._series.aligned_matrix(ref_rows)
            n_reflect_untracked = sum(1 for r in ref_rows if r is None)
            dim, n_ref, rel = nm.dim_map(duty, swing, self._base, dyn, lab, cand,
                                         ref_series, nm.NEIGH_R, nm.DIM_DYNAMIC,
                                         nm.DIM_REFLECT)

            dyn_pts = nm.dyn_samples(duty, dyn)
            per_rows = ([self._tracked_row(y, x) for y, x in fpts]
                        + [self._tracked_row(y, x) for y, x in dyn_pts])
            per_data = [self._series.series(r) for r in per_rows]
            n_period_untracked = sum(1 for r in per_rows if r is None)

            def flashing(i, y, x):
                on = (per_data[i].astype(np.float64) - self._base[y, x]) >= nm.ON_DELTA
                if onsets_of(on) < self._period.onset_min:
                    return False
                curve = ac_limited(on, lag_lo, lag_hi)[2]
                return clean_train(curve, lag_lo, self._period.peak_floor,
                                   self._period.min_peaks,
                                   self._period.gap_tol)[0]

            flash_u8 = np.zeros((self._h, self._w), np.uint8)
            for i, c in enumerate(cand):
                if flashing(i, *fpts[i]):
                    ys, xs = cand_px[i]
                    flash_u8[ys, xs] = 1
                    r = max(stats[c, cv.CC_STAT_WIDTH],
                            stats[c, cv.CC_STAT_HEIGHT]) // 2 + nm.RING_PAD
                    rings.append((fpts[i][1], fpts[i][0], r))
                    n_flash += 1
            for j, (y, x) in enumerate(dyn_pts):
                if flashing(len(cand) + j, y, x):
                    cv.circle(flash_u8, (x, y), nm.FLASH_R, 1, -1)
                    rings.append((x, y, nm.RING_R))
                    n_flash += 1
            flash = flash_u8 > 0
            if report_type == R.REPORT_TYPE_BURST:
                self._accumulate_flash(flash)

        ref = dim == nm.DIM_REFLECT
        dim[flash & ~ref] = 1.0
        gate = np.maximum(flash.astype(np.float32), nm.steady_gate(duty))
        keep = nm.despeckle(duty, nm.DESPECKLE_FLOOR, nm.DESPECKLE_K,
                            nm.DESPECKLE_MIN)
        heat = nm.heat_bgr(duty, top, nm.RENDER_GAMMA, nm.CMAPS["ice"])
        heat = heat * dim[..., None]
        alpha = nm.alpha_of(duty, keep, nm.RENDER_DUTY_LO, nm.BLEND)
        vis = nm.compose(heat, alpha * (dim * gate)[..., None], nm.night_bg(self._bg))
        nm.draw_rings(vis, rings)
        nm.draw_bar(vis, nm.CMAPS["ice"], "flash duty")

        image = vis
        image_mode = R.IMAGE_MODE_NIGHT
        day_align = None
        if base_frame_bgr is None:
            alignment_status = AlignmentStatus.BASE_UNAVAILABLE
        else:
            day_bgr, dx, dy, cc = nm.align_day(base_frame_bgr,
                                               np.clip(self._bg, 0, 255))
            alignment_status = R.alignment_status_for(cc, self._alignment)
            day_align = {"dx": round(dx, 2), "dy": round(dy, 2),
                         "cc": round(cc, 4)}
            if alignment_status != AlignmentStatus.REJECTED:
                dheat = nm.heat_bgr(duty, top, nm.RENDER_GAMMA,
                                    nm.CMAPS["inferno"])
                dheat = dheat * dim[..., None]
                dalpha = (nm.alpha_ramp(duty, keep, nm.DAY_LO, nm.DAY_HI,
                                        nm.DAY_ALPHA)
                          * (dim * gate)[..., None])
                dvis = nm.compose(dheat, dalpha,
                                  nm.mute_bg(day_bgr, nm.DAY_BG_SAT,
                                             nm.DAY_BG_DARK))
                nm.draw_rings(dvis, rings)
                nm.draw_bar(dvis, nm.CMAPS["inferno"], "flash duty")
                dvis = nm.draw_caption(
                    dvis, "camera %s  duty>=%.2f  delta=%d  p99=%.3f"
                    % (self._camera_id, nm.DAY_LO, nm.ON_DELTA, top))
                image = dvis
                image_mode = R.IMAGE_MODE_DAY

        ok, buf = cv.imencode(".jpg", image,
                              [int(cv.IMWRITE_JPEG_QUALITY), R.JPEG_QUALITY])
        jpeg = buf.tobytes() if ok else None

        status = R.status_for_alignment(R.STATUS_OK, alignment_status)

        meta = {
            R.M_ALIGNMENT: alignment_status.value,
            R.M_CANDIDATE_OVERFLOW: bool(overflow),
            R.M_SAMPLING: self._sampling_metadata(),
            R.M_N_CAND: len(cand),
            R.M_N_CAND_TOTAL: n_cand_total,
            R.M_N_FLASH: int(n_flash),
            R.M_N_REFLECT: int(n_ref),
            R.M_FLASH_SEMANTICS: (R.FLASH_SEMANTICS_PRESENCE if use_union
                                  else R.FLASH_SEMANTICS_BURST),
            R.M_FLASH_AREA: int(flash.sum()),
            R.M_FLASH_UNION_AREA: (int(self._flash_union.sum())
                                   if self._flash_union is not None
                                   else int(flash.sum())),
            R.M_FLASH_BUCKET_COUNT: int(self._n_flash_buckets),
            R.M_CALIBRATION_STATUS: getattr(self._spec.status, "value",
                                            self._spec.status),
            R.M_LAMP_ROI: {"enabled": bool(self._lamp_roi),
                           "frac": self._lamp_roi_frac},
            R.M_ONLINE: {
                "n_tracked_points": int(self._series.n_points),
                "n_tracked_candidates": int(self._discovery.n_candidate_cores),
                "n_tracked_dyn": int(self._discovery.n_dyn_cores),
                "n_discovery_passes": int(self._discovery.n_passes),
                "first_discovery_sample": self._discovery.first_discovery_sample,
                "n_components_seen": int(self._discovery.n_components_seen),
                "n_candidates_dropped_online": int(self._discovery.n_cand_dropped),
                "n_dyn_dropped_online": int(self._discovery.n_dyn_dropped),
                "n_dropped_online_total": int(self._discovery.n_dropped),
                "n_candidates_dropped_final": int(n_cand_dropped_final),
                "n_candidates_kept_final": int(len(cand)),
                "n_reflect_untracked": int(n_reflect_untracked),
                "n_period_untracked": int(n_period_untracked),
            },
            R.M_DAY_ALIGN: day_align,
            R.M_IMAGE_MODE: image_mode,
            R.M_NIGHT_MED: round(float(self._detrend), 2),
            R.M_DELTA: int(nm.ON_DELTA),
            R.M_LAG_FRAMES: [lag_lo, lag_hi],
            R.M_PERIOD_STEP: int(self._period.period_step),
            R.M_WARMUP: {
                "frames": int(self._baseline_frames),
                "warmup_s": int(self._warmup_s),
                "released": self._warm_stack is None,
            },
            R.M_MEMORY_MB: self._memory_estimate_mb(),
            R.M_BURST: self._burst_metadata(),
            R.M_RESTORED: bool(self._restored),
        }
        meta.update(self._night_meta(night))
        return R.build_report(self._camera_id, created_at, jpeg, meta,
                              status=status, report_type=report_type)

    @staticmethod
    def _iso(value):
        if value is None:
            return None
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def _burst_metadata(self):
        return {
            "overnight": self._overnight,
            "cadence_minutes": self._cadence_minutes,
            "burst_seconds": self._burst_seconds,
            "started_at": self._iso(self._burst_started_wall),
            "ended_at": self._iso(self._burst_ended_wall),
            "samples": int(self._burst_n),
            "design_rate_hz": round(self._design_rate_hz(), 4),
            "effective_rate_hz": round(self._rate_hz(), 4),
        }

    @_synchronized
    def release(self) -> None:
        self._warm_stack = None
        self._warm_meds = None
        self._base = None
        self._bg = None
        self._on = None
        self._peak = None
        self._excess = None
        self._on_hit = None
        self._mask = None
        self._flash_union = None
        self._flash_count = None
        self._n_flash_buckets = 0
        self._last_med = None
        self._base_reset = False
        if self._series is not None:
            self._series.release()
        if self._discovery is not None:
            self._discovery.release()
        self._released = True
