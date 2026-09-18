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
* ``max_candidates`` overflow sets ``candidate_overflow`` (never silent) and
  is surfaced as ``Report.status == "degraded"``.
* ``series_cap is None`` keeps the full sequence (production default).
  A non-None cap keeps only the tail window and is marked experimental.
* ``release()`` drops every accumulator so a failed/hung finalize cannot leak
  into the next night.

Known online-vs-golden differences (reported in metadata, not hidden):
  - warm baseline = first N samples, golden = N spread over the whole night;
  - series start at discovery time, golden starts at sample 0;
  - lag windows use the measured sample rate, golden uses video fps/step;
  - reflection hosts are matched to the nearest tracked pixel;
  - dynamic-blob lamp cores are only tracked if discovered while rolling.
"""
from __future__ import annotations

import os
import sys

import cv2 as cv
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.contracts import AlignmentStatus, NightLampSpec  # noqa: E402

from night_lamp import report as R  # noqa: E402
from night_lamp.periodicity import ac_limited, clean_train, onsets_of  # noqa: E402
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
        self._max_candidates = int(spec.memory.max_candidates)
        self._series_cap = spec.memory.series_cap
        max_points = 3 * self._max_candidates + nm.DYN_MAX_PEAK
        self._series = TemporalSeries(max_points, cap=self._series_cap)
        self._discovery = CandidateDiscovery(self._series, self._max_candidates)
        self._baseline_frames = max(1, int(spec.baseline.frames))
        self._warmup_s = float(spec.baseline.warmup_s)
        self._period = spec.periodicity
        self._alignment = spec.alignment
        self._discovery_every = _DISCOVERY_EVERY

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

        self._meds = []
        self._n_seen = 0
        self._first_ts = None
        self._last_ts = None
        self._frozen = False
        self._released = False
        self._dropped_after_freeze = 0

    @property
    def sample_count(self) -> int:
        return self._n_seen

    @property
    def candidate_overflow(self) -> bool:
        return self._discovery.overflow

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
        if self._first_ts is None:
            self._first_ts = float(ts_mono)
        self._last_ts = float(ts_mono)
        med = float(np.median(gray))
        self._n_seen += 1
        self._meds.append(med)

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

        self._process(gray, med)
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

    def _process(self, gray, med):
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
        self._series.append(gray, self._mask, off)

    def _run_discovery(self):
        if not self._warm_done or self._n_seen <= 0 or self._on is None:
            return
        duty = self._on / float(self._n_seen)
        self._discovery.observe(duty, self._on, self._n_seen, self._peak)

    def freeze(self) -> None:
        if self._released:
            raise RuntimeError("NightSession is released")
        if self._frozen:
            return
        if not self._warm_done and self._n_seen > 0:
            self._finish_warmup()
        self._frozen = True
        self._run_discovery()

    def _rate_hz(self):
        if self._first_ts is None or self._last_ts is None:
            return 0.0
        span = self._last_ts - self._first_ts
        if span <= 0:
            return 0.0
        return (self._n_seen - 1) / span

    def _sampling_metadata(self):
        span = (0.0 if self._first_ts is None or self._last_ts is None
                else self._last_ts - self._first_ts)
        return {
            "interval_ms": int(self._spec.sampling.interval_ms),
            "actual_samples": int(self._n_seen),
            "actual_span_s": round(span, 3),
            "actual_rate_hz": round(self._rate_hz(), 4),
            "mode": "tail" if self._series_cap is not None else "full",
            "series_cap": self._series_cap,
            "experimental_tail_window": self._series_cap is not None,
        }

    def _memory_estimate_mb(self):
        total = 0
        if self._series is not None:
            total += self._series.memory_bytes()
        for a in (self._base, self._bg, self._on, self._peak, self._mask):
            if a is not None:
                total += int(a.nbytes)
        if self._meds:
            total += len(self._meds) * 8
        return round(total / 1048576.0, 1)

    def _night_qualification(self):
        """Replay ``NightLampSpec.night_gate`` over the accumulated medians.

        This is a pure *consumer* of state the session already holds: it reads
        ``self._meds`` (the global per-sample medians) and applies the exact
        hysteresis rule of ``night_state.update`` (enter_threshold <
        exit_threshold, ``persistence`` consecutive dark samples to enter, one
        bright sample above exit to leave).  It never drops samples and never
        feeds back into the nightly map maths, so the golden/online numerics
        are unchanged -- it only labels the night for the report.
        """
        gate = self._spec.night_gate or {}
        meds = self._meds or []
        info = {
            "configured": False,
            "valid": False,
            "enter_threshold": gate.get("enter_threshold"),
            "exit_threshold": gate.get("exit_threshold"),
            "persistence": gate.get("persistence"),
            "samples": len(meds),
            "transitions": 0,
        }
        t_enter = gate.get("enter_threshold")
        t_exit = gate.get("exit_threshold")
        if t_enter is None or t_exit is None:
            info["state"] = "not_configured"
            info["qualified"] = None
            return info
        try:
            t_enter = float(t_enter)
            t_exit = float(t_exit)
            persistence = max(1, int(gate.get("persistence", 1)))
        except (TypeError, ValueError):
            info["state"] = "invalid_gate"
            info["qualified"] = None
            return info
        info["enter_threshold"] = t_enter
        info["exit_threshold"] = t_exit
        info["persistence"] = persistence
        info["configured"] = True
        if t_enter >= t_exit:
            info["state"] = "invalid_gate"
            info["qualified"] = None
            info["reason"] = "requires enter_threshold < exit_threshold"
            return info
        info["valid"] = True
        state, count, transitions = NS.TWILIGHT, 0, 0
        for g in meds:
            prev = state
            state, count, _entered = NS.update(
                state, count, float(g), t_enter, t_exit, persistence)
            if state != prev:
                transitions += 1
        info["state"] = state
        info["qualified"] = (state == NS.NIGHT)
        info["transitions"] = transitions
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
                R.M_CANDIDATE_OVERFLOW: self._discovery.overflow,
                R.M_SAMPLING: self._sampling_metadata(),
                R.M_REASON: "no_samples",
            }
            meta.update(self._night_meta(night))
            return R.build_report(self._camera_id, created_at, None, meta,
                                  status=R.STATUS_FAILED)
        try:
            return self._finalize_inner(base_frame_bgr, created_at, night)
        except Exception as exc:  # never raise into the runtime loop
            meta = {
                R.M_ALIGNMENT: AlignmentStatus.REJECTED.value,
                R.M_CANDIDATE_OVERFLOW: self._discovery.overflow,
                R.M_SAMPLING: self._sampling_metadata(),
                R.M_REASON: "%s: %s" % (type(exc).__name__, exc),
            }
            meta.update(self._night_meta(night))
            return R.build_report(self._camera_id, created_at, None, meta,
                                  status=R.STATUS_FAILED)

    def _finalize_inner(self, base_frame_bgr, created_at, night):
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
        overflow = (self._discovery.overflow
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

        rate = self._rate_hz()
        if rate <= 0:
            rate = 1.0
        lag_lo = max(1, int(round(self._period.lag_lo_s * rate)))
        lag_hi = max(lag_lo + 1, int(round(self._period.lag_hi_s * rate)))

        def flashing(i, y, x):
            on = (per_data[i].astype(np.float64) - self._base[y, x]) >= nm.ON_DELTA
            if onsets_of(on) < self._period.onset_min:
                return False
            curve = ac_limited(on, lag_lo, lag_hi)[2]
            return clean_train(curve, lag_lo, self._period.peak_floor,
                               self._period.min_peaks, self._period.gap_tol)[0]

        rings = []
        flash = np.zeros((self._h, self._w), np.uint8)
        n_flash = 0
        for i, c in enumerate(cand):
            if flashing(i, *fpts[i]):
                ys, xs = cand_px[i]
                flash[ys, xs] = 1
                r = max(stats[c, cv.CC_STAT_WIDTH],
                        stats[c, cv.CC_STAT_HEIGHT]) // 2 + nm.RING_PAD
                rings.append((fpts[i][1], fpts[i][0], r))
                n_flash += 1
        for j, (y, x) in enumerate(dyn_pts):
            if flashing(len(cand) + j, y, x):
                cv.circle(flash, (x, y), nm.FLASH_R, 1, -1)
                rings.append((x, y, nm.RING_R))
                n_flash += 1
        flash = flash > 0

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

        status = R.STATUS_DEGRADED if overflow else R.STATUS_OK
        status = R.status_for_alignment(status, alignment_status)

        meta = {
            R.M_ALIGNMENT: alignment_status.value,
            R.M_CANDIDATE_OVERFLOW: bool(overflow),
            R.M_SAMPLING: self._sampling_metadata(),
            R.M_N_CAND: len(cand),
            R.M_N_CAND_TOTAL: n_cand_total,
            R.M_N_FLASH: int(n_flash),
            R.M_N_REFLECT: int(n_ref),
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
        }
        meta.update(self._night_meta(night))
        return R.build_report(self._camera_id, created_at, jpeg, meta,
                              status=status)

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
        self._meds = None
        if self._series is not None:
            self._series.release()
        if self._discovery is not None:
            self._discovery.release()
        self._released = True
