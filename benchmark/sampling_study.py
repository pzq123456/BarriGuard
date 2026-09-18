"""Overnight Sampling Study: fidelity/cost of observing less than the full night.

Question
--------
If the morning report is "one night heatmap + flasher set", do we need 6.25 Hz
for the whole night, or is an hourly burst enough?  This harness measures the
cost/fidelity curve; it does not pick a policy.

Method (benchmark-only; production is untouched)
-----------------------------------------------
1. Simulate the production interval gate over a clip -> the full-rate accepted
   observation set (the REFERENCE schedule).
2. For each policy (burst / random / sparse) take a subset of those accepted
   observations.
3. Feed exactly that subset into a fresh production ``NightSession`` with a
   shared baseline injected (so spatial duty differences reflect only *which*
   observations were taken, not baseline drift).
4. Compare against the reference on:
   * spatial  : duty correlation + hotspot IoU over the valid (non-OSD) region;
   * temporal : per-target flasher agreement (pos/neg), isolated into two
                policies --
                  - PERSISTENT: one session for the whole schedule;
                  - RESET     : a new temporal state after an observation gap
                                (> --reset-gap-s), flashers = union over bursts.
5. Report observations, wall time, ms/obs, peak RSS.

Limitations (explicit)
----------------------
* The clip is a MINI-NIGHT (<= 35 min); 7 h conclusions are extrapolation and
  must be marked as such.  No full-night recording exists in tmp/.
* Baseline is frozen from the reference warmup and injected into every policy;
  per-burst warmup is therefore not modelled.
* Reference is the full-rate session on the same clip, not ground truth.

Usage
-----
    python benchmark/sampling_study.py --max-min 3 --policies continuous,burst:300:30
    python benchmark/sampling_study.py --policies continuous,burst:300:30,burst:300:60,random:0.1
    python benchmark/sampling_study.py            # full 35 min mini-night
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from server.contracts import (  # noqa: E402
    AlignmentSpec, AlignmentStatus, BaselineSpec, MemorySpec, NightLampSpec,
    PeriodicitySpec, SamplingSpec,
)
from night_lamp import session as S  # noqa: E402
from night_lamp.periodicity import ac_limited, clean_train, onsets_of  # noqa: E402
from night_lamp.tools import nightly_map as nm  # noqa: E402

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

DEFAULT_POLICIES = ("continuous,burst:300:30,burst:300:60,burst:300:120,"
                    "burst:600:60,burst:600:120,random:0.1,sparse:4:120")


# --------------------------------------------------------------------------- spec
def load_spec(cfg_path: Path, camera_id: str) -> NightLampSpec:
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cam = next(c for c in d["cameras"] if c["id"] == camera_id)
    nl = cam["algorithms"]["night_lamp"]
    s = nl.get("sampling", {}) or {}
    m = nl.get("memory", {}) or {}
    b = nl.get("baseline", {}) or {}
    p = nl.get("periodicity", {}) or {}
    a = nl.get("alignment", {}) or {}
    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=int(s.get("interval_ms", 160))),
        memory=MemorySpec(max_candidates=int(m.get("max_candidates", 200)),
                          series_cap=m.get("series_cap", None)),
        baseline=BaselineSpec(frames=int(b.get("frames", 60)),
                              warmup_s=int(b.get("warmup_s", 600))),
        periodicity=PeriodicitySpec(
            period_step=int(p.get("period_step", 2)),
            lag_lo_s=float(p.get("lag_lo_s", 0.3)),
            lag_hi_s=float(p.get("lag_hi_s", 7.0)),
            peak_floor=float(p.get("peak_floor", 0.2)),
            min_peaks=int(p.get("min_peaks", 3)),
            gap_tol=int(p.get("gap_tol", 1)),
            onset_min=int(p.get("onset_min", 50))),
        alignment=AlignmentSpec(method=a.get("method", "ecc_translation"),
                                min_cc=float(a.get("min_cc", 0.5)),
                                on_low_cc=AlignmentStatus(a.get("on_low_cc", "degraded"))),
    )


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 1048576.0


# --------------------------------------------------------------------------- gate
def accepted_obs(n_frames: int, fps: float, interval_s: float):
    """Production gate replay without decoding: [(frame_idx, wall_t), ...]."""
    out, last = [], None
    for f in range(n_frames):
        t = f / fps
        if last is None or (t - last) >= interval_s:
            out.append((f, t))
            last = t
    return out


def policy_mask(name: str, times: list[float], total_s: float, seed: int) -> np.ndarray:
    """Boolean mask over the accepted list for a policy string."""
    t = np.asarray(times, np.float64)
    if name == "continuous":
        return np.ones(t.size, bool)
    if name.startswith("burst:"):
        _, P, W = name.split(":")
        P, W = float(P), float(W)
        return (np.mod(t, P) < W)
    if name.startswith("random:"):
        frac = float(name.split(":")[1])
        rng = np.random.default_rng(seed)
        return rng.random(t.size) < frac
    if name.startswith("sparse:"):
        _, n, W = name.split(":")
        n, W = int(n), float(W)
        m = np.zeros(t.size, bool)
        for k in range(n):
            c = total_s * (k + 0.5) / n
            m |= (t >= c - W / 2.0) & (t < c + W / 2.0)
        return m
    raise ValueError(f"unknown policy: {name!r}")


# --------------------------------------------------------------------------- session
def capture_baseline(sess) -> dict:
    return {"mask": sess._mask, "base": sess._base.copy(),
            "bg": sess._bg.copy(), "detrend": sess._detrend,
            "h": sess._h, "w": sess._w}


def new_session(camera: str, spec: NightLampSpec, bl: dict):
    """Production session with the reference baseline injected (no warmup)."""
    s = S.NightSession(camera, spec)
    s._h, s._w = bl["h"], bl["w"]
    s._mask = bl["mask"]
    s._detrend = bl["detrend"]
    s._base = bl["base"].copy()
    s._bg = bl["bg"].copy()
    s._on = np.zeros_like(bl["base"])
    s._peak = np.zeros_like(bl["base"])
    s._warm_done = True
    return s


def flashing(series: np.ndarray, base_val: float, lag_lo: int, lag_hi: int,
             period: PeriodicitySpec):
    """Exact production flash rule; returns (bool, peak_lag)."""
    on = (series.astype(np.float64) - base_val) >= nm.ON_DELTA
    if onsets_of(on) < period.onset_min:
        return False, -1
    curve = ac_limited(on, lag_lo, lag_hi)[2]
    if curve.size == 0:
        return False, -1
    ok = clean_train(curve, lag_lo, period.peak_floor,
                     period.min_peaks, period.gap_tol)[0]
    return ok, lag_lo + int(np.argmax(curve))


def decide(sess, y: int, x: int, base, lag_lo: int, lag_hi: int,
           period: PeriodicitySpec):
    """Decision for one target in one session; (False, -1) when untracked."""
    if sess is None:
        return False, -1
    r = sess._series.lookup(int(y), int(x))
    if r is None or not sess._series.has_data(r):
        return False, -1
    s = sess._series.series(r)
    if s.size < 4:
        return False, -1
    return flashing(s, float(base[y, x]), lag_lo, lag_hi, period)


# --------------------------------------------------------------------------- reference
def run_reference(video: Path, spec: NightLampSpec, camera: str,
                  acc_frames, interval_s: float, total_frames: int):
    """Full-rate reference session + duty/peak + flash targets."""
    ref = S.NightSession(camera, spec)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    it = 0
    t_wall = time.perf_counter()
    rss0 = _rss_mb()
    rss_peak = rss0
    for f in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if it < len(acc_frames) and f == acc_frames[it]:
            ref.accumulate(frame, it * interval_s)
            it += 1
            if it % 5000 == 0:
                rss_peak = max(rss_peak, _rss_mb())
    cap.release()
    ref.freeze()
    wall = time.perf_counter() - t_wall
    return {"session": ref, "n_obs": it, "wall_s": round(wall, 2),
            "rss_peak_mb": round(max(rss_peak, _rss_mb()), 1)}


def build_targets(ref, spec: NightLampSpec, period: PeriodicitySpec,
                  lag_lo: int, lag_hi: int):
    """Authoritative candidates + dyn cores, and the reference flasher set."""
    duty = ref._on / float(ref._n_seen)
    dyn, lab, stats, cand_all = nm.split_comps(duty)
    cand_all = sorted(cand_all, key=lambda c: -stats[c, cv.CC_STAT_AREA])
    cand = cand_all[:spec.memory.max_candidates]

    targets = []
    for c in cand:
        ys, xs = np.nonzero(lab == c)
        if ys.size == 0:
            continue
        jf = int(np.argmax(duty[ys, xs]))
        targets.append(("cand", int(ys[jf]), int(xs[jf])))
    for (y, x) in nm.dyn_samples(duty, dyn):
        targets.append(("dyn", int(y), int(x)))

    ref_flash, ref_lag = [], []
    for _k, y, x in targets:
        ok, lag = decide(ref, y, x, ref._base, lag_lo, lag_hi, period)
        ref_flash.append(ok)
        ref_lag.append(lag)
    return {
        "duty": duty.astype(np.float32),
        "dyn": dyn,
        "n_cand_all": len(cand_all),
        "n_cand_used": len(cand),
        "targets": targets,
        "ref_flash": np.asarray(ref_flash, bool),
        "ref_lag": np.asarray(ref_lag, np.int32),
    }


# --------------------------------------------------------------------------- policy pass
class PolicyState:
    """One policy: a persistent session plus gap-reset burst sessions.

    Both run side by side on the same accepted observations.  The persistent
    session concatenates bursts (cross-gap discontinuity included); the burst
    sessions start fresh after any gap > ``reset_gap_s``.
    """

    def __init__(self, name, spec, camera, bl):
        self.name = name
        self._spec = spec
        self._camera = camera
        self._bl = bl
        self.persistent = new_session(camera, spec, bl)
        self.bursts: list = []
        self._cur = None
        self._cur_i = 0
        self._last_ord = None
        self.n_obs = 0

    def feed(self, frame, ordinal: int, interval_s: float, reset_gap_s: float):
        if self._cur is None or (self._last_ord is not None and
                                 (ordinal - self._last_ord) * interval_s > reset_gap_s):
            if self._cur is not None:
                self.bursts.append(self._cur)
            self._cur = new_session(self._camera, self._spec, self._bl)
            self._cur_i = 0
        self.persistent.accumulate(frame, self.n_obs * interval_s)
        self._cur.accumulate(frame, self._cur_i * interval_s)
        self._cur_i += 1
        self.n_obs += 1
        self._last_ord = ordinal

    def finish(self):
        if self._cur is not None:
            self.bursts.append(self._cur)
        self.persistent.freeze()


def run_all_policies(video: Path, spec, camera: str, bl: dict,
                     acc_frames, masks: dict, interval_s: float,
                     total_frames: int, reset_gap_s: float):
    """Single decode pass feeding every policy state in parallel."""
    states = {name: PolicyState(name, spec, camera, bl) for name in masks}
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    it = 0
    t_wall = time.perf_counter()
    rss0 = _rss_mb()
    rss_peak = rss0
    for f in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if it < len(acc_frames) and f == acc_frames[it]:
            for st in states.values():
                if masks[st.name][it]:
                    st.feed(frame, it, interval_s, reset_gap_s)
            it += 1
            if it % 4000 == 0:
                rss_peak = max(rss_peak, _rss_mb())
    cap.release()
    for st in states.values():
        st.finish()
    wall = time.perf_counter() - t_wall
    return states, {"wall_s": round(wall, 2),
                    "rss_peak_mb": round(max(rss_peak, _rss_mb()), 1)}


def temporal_metrics(persistent, bursts, targets, ref_flash, ref_lag,
                     base, lag_lo, lag_hi, period):
    """TP/FP/FN for persistent and gap-reset, plus peak-lag deviation."""
    def score(sess):
        got = []
        for _k, (_kind, y, x) in enumerate(targets):
            ok, _lag = decide(sess, y, x, base, lag_lo, lag_hi, period)
            got.append(ok)
        g = np.asarray(got, bool)
        tp = int(np.count_nonzero(g & ref_flash))
        fp = int(np.count_nonzero(g & ~ref_flash))
        fn = int(np.count_nonzero(~g & ref_flash))
        return g, tp, fp, fn

    def score_reset():
        got = np.zeros(len(targets), bool)
        for b in bursts:
            for k, (_kind, y, x) in enumerate(targets):
                if got[k]:
                    continue
                ok, _lag = decide(b, y, x, base, lag_lo, lag_hi, period)
                if ok:
                    got[k] = True
        tp = int(np.count_nonzero(got & ref_flash))
        fp = int(np.count_nonzero(got & ~ref_flash))
        fn = int(np.count_nonzero(~got & ref_flash))
        return got, tp, fp, fn

    def lag_dev(sess):
        devs = []
        for k, (_kind, y, x) in enumerate(targets):
            if not ref_flash[k] or ref_lag[k] < 0:
                continue
            ok, lag = decide(sess, y, x, base, lag_lo, lag_hi, period)
            if ok and lag >= 0:
                devs.append(abs(lag - ref_lag[k]))
        if not devs:
            return None
        return round(float(np.median(devs)), 2)

    p_got, p_tp, p_fp, p_fn = score(persistent)
    r_got, r_tp, r_fp, r_fn = score_reset()
    return {
        "n_targets": len(targets),
        "ref_flashers": int(ref_flash.sum()),
        "persistent": {"tp": p_tp, "fp": p_fp, "fn": p_fn,
                       "recall": _ratio(p_tp, p_tp + p_fn),
                       "precision": _ratio(p_tp, p_tp + p_fp),
                       "median_lag_dev_frames": lag_dev(persistent)},
        "reset": {"tp": r_tp, "fp": r_fp, "fn": r_fn,
                  "recall": _ratio(r_tp, r_tp + r_fn),
                  "precision": _ratio(r_tp, r_tp + r_fp)},
        "reset_n_bursts": len(bursts),
    }


def spatial_metrics(duty_ref, duty_pol, mask):
    valid = ~mask
    a = duty_ref[valid].astype(np.float64)
    b = duty_pol[valid].astype(np.float64)
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        corr = None
    else:
        corr = round(float(np.corrcoef(a, b)[0, 1]), 4)
    thr = float(np.percentile(a, 99.0))
    ref_hot = a > thr
    pol_hot = b > thr
    union = int(np.count_nonzero(ref_hot | pol_hot))
    iou = round(int(np.count_nonzero(ref_hot & pol_hot)) / union, 4) if union else None
    return {"duty_corr": corr, "hotspot_iou_p99": iou,
            "hotspot_thr": round(thr, 4)}


def _ratio(a: int, b: int):
    return round(a / b, 4) if b else None


# --------------------------------------------------------------------------- driver
def run(args) -> dict:
    spec = load_spec(Path(args.config), args.camera)
    interval_s = spec.sampling.interval_ms / 1000.0
    video = Path(args.video)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    total_frames = int(round(args.max_min * 60.0 * fps))
    cap.release()
    total_s = total_frames / fps

    acc = accepted_obs(total_frames, fps, interval_s)
    acc_frames = [f for f, _ in acc]
    acc_times = [t for _, t in acc]
    period = spec.periodicity
    rate = 1.0 / interval_s
    lag_lo = max(1, int(round(period.lag_lo_s * rate)))
    lag_hi = max(lag_lo + 1, int(round(period.lag_hi_s * rate)))

    print(f"[input] {video.name} fps={fps:.3f} capped={args.max_min}min "
          f"frames={total_frames} accepted={len(acc)} rate={rate:.2f}Hz "
          f"lag=[{lag_lo},{lag_hi}]")
    base_rss = _rss_mb()

    print("[reference] full-rate ...")
    ref_run = run_reference(video, spec, args.camera, acc_frames,
                            interval_s, total_frames)
    ref = ref_run["session"]
    bl = capture_baseline(ref)
    tgt = build_targets(ref, spec, period, lag_lo, lag_hi)
    print(f"[reference] obs={ref_run['n_obs']} wall={ref_run['wall_s']}s "
          f"targets={len(tgt['targets'])} ref_flashers={int(tgt['ref_flash'].sum())} "
          f"cand={tgt['n_cand_used']}/{tgt['n_cand_all']} "
          f"rss_peak={ref_run['rss_peak_mb']}MB")

    policies = [p for p in args.policies.split(",") if p]
    masks = {name: policy_mask(name, acc_times, total_s, args.random_seed)
             for name in policies}
    for name in policies:
        print(f"[policy] {name}: accepted={int(masks[name].sum())} "
              f"({int(masks[name].sum()) / max(len(acc), 1):.1%})")
    print("[policy-pass] decoding once for all policies ...")
    states, pass_meta = run_all_policies(
        video, spec, args.camera, bl, acc_frames, masks,
        interval_s, total_frames, args.reset_gap_s)
    ref_ms = ref_run["wall_s"] * 1000.0 / max(ref_run["n_obs"], 1)

    rows = []
    for name in policies:
        st = states[name]
        pers = st.persistent
        duty_pol = (pers._on / float(pers._n_seen)) if pers._n_seen else None
        row = {
            "policy": name,
            "n_obs": st.n_obs,
            "obs_frac": round(st.n_obs / max(len(acc), 1), 4),
            "proj_wall_s": round(st.n_obs * ref_ms / 1000.0, 1),
            "ms_per_obs_ref": round(ref_ms, 2),
            "series_mb": round(pers._series.memory_bytes() / 1048576.0, 2),
            "n_tracked_points": int(pers._series.n_points),
            "candidate_overflow": bool(pers._discovery.overflow),
        }
        if duty_pol is None:
            row["spatial"] = {"duty_corr": None, "hotspot_iou_p99": None,
                              "hotspot_thr": None}
            row["temporal"] = None
        else:
            row["spatial"] = spatial_metrics(tgt["duty"], duty_pol, bl["mask"])
            row["temporal"] = temporal_metrics(
                pers, st.bursts, tgt["targets"], tgt["ref_flash"],
                tgt["ref_lag"], bl["base"], lag_lo, lag_hi, period)
        rows.append(row)
        print(f"  -> {name}: spatial={row['spatial']} "
              f"P={row['temporal']['persistent'] if row['temporal'] else None} "
              f"R={row['temporal']['reset'] if row['temporal'] else None}")

    return {
        "mode": "sampling_study",
        "video": str(video.resolve()),
        "camera": args.camera,
        "fps": round(fps, 4),
        "capped_min": args.max_min,
        "interval_ms": spec.sampling.interval_ms,
        "accepted_reference": len(acc),
        "reset_gap_s": args.reset_gap_s,
        "base_rss_mb": round(base_rss, 1),
        "reference": {"n_obs": ref_run["n_obs"], "wall_s": ref_run["wall_s"],
                      "ms_per_obs": round(ref_ms, 2),
                      "targets": len(tgt["targets"]),
                      "ref_flashers": int(tgt["ref_flash"].sum()),
                      "n_cand_used": tgt["n_cand_used"],
                      "n_cand_all": tgt["n_cand_all"]},
        "policy_pass": pass_meta,
        "policies": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Overnight sampling fidelity/cost study")
    ap.add_argument("--video", default="tmp/1750_20260913_220000.mp4")
    ap.add_argument("--config", default="deploy/config.yaml")
    ap.add_argument("--camera", default="1750")
    ap.add_argument("--max-min", type=float, default=35.0,
                    help="cap on clip length in minutes (mini-night)")
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--reset-gap-s", type=float, default=30.0,
                    help="observation gap that starts a new temporal burst")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark/out")
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    r = run(a)
    fp = outdir / f"sampling_study_{a.camera}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    fp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== summary ===")
    print(f"{'policy':22s} {'obs%':>6s} {'corr':>7s} {'IoU':>6s} "
          f"{'P.recall':>8s} {'P.prec':>7s} {'R.recall':>8s} {'R.prec':>7s} "
          f"{'proj_s':>7s}")
    for row in r["policies"]:
        sp = row["spatial"]
        tm = row["temporal"]
        pp = tm["persistent"] if tm else {}
        rr = tm["reset"] if tm else {}
        print(f"{row['policy']:22s} {row['obs_frac']:6.1%} "
              f"{_s(sp['duty_corr'],''):>7s} {_s(sp['hotspot_iou_p99'],''):>6s} "
              f"{_s(pp.get('recall')):>8s} {_s(pp.get('precision')):>7s} "
              f"{_s(rr.get('recall')):>8s} {_s(rr.get('precision')):>7s} "
              f"{row['proj_wall_s']:7.1f}")
    print(f"policy-pass wall={r['policy_pass']['wall_s']}s "
          f"rss_peak={r['policy_pass']['rss_peak_mb']}MB")
    print(f"saved {fp}")
    return 0


def _s(v, dash="-"):
    return dash if v is None else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
