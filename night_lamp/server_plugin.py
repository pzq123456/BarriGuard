"""Night-lamp server adapter (black box): server feeds every frame, we own the rest.

server sees an ordinary per_frame Algorithm (name/cadence/step/reset); all
burst/night/evidence logic from night_lamp/main.py lives inside, adapted from
burst_at() file semantics to step() stream semantics:

- burst assembly: first 40 accepted frames after due time (~3.2s @12.49fps).
  Duplicates dropped by 48x27 signature (worker may re-feed the latest frame);
  arrival gaps recorded, over-budget bursts marked degraded and skipped
  (counters frozen, like main.py's failed bursts).
- cadence: evaluate once per burst.interval_sec (production 3600; stress via
  env NIGHT_LAMP_BURST_S, e.g. 300 tonight). Non-burst steps only refresh
  cheap per-frame signals and return cached annots.
- night gate: identical NS.update on burst global_median; non-NIGHT bursts
  skip lamp eval (record globals only).
- evidence: lamp/global jsonl + bursts/*.json + raw/overlay snapshot per
  evaluated burst (hourly in production), rooted at calib evidence.out.
  Mid-interval snapshots don't exist live (no seeking) -- dropped by design.
- P0 alert machine (provisional, M07-OLD dead / M07-NEW alive 2026-09-11):
  alive evidence (F2==1 or S==1) resets dead_count; dead evidence
  (F2==0 and swing<floor and S==0) increments; borderline (F2==0 but
  swing>=floor) resets silently. count==1 -> suspected, count>=2 -> alarm
  (every NIGHT burst while persisting). GATED/degraded/failed bursts change
  nothing. Steady lamps ride the same machine (alive S==1; dead steady is
  flat+dark -> candidate; flat+bright+allON is S==1 -> alive).
- step() never raises (worker loop must survive us): eval exceptions become
  errors.jsonl rows and failed bursts.

Memory: 40 gray frames (~83MB) transient per burst + 1 BGR for snapshot.
Per-step cost: one gray convert + downsampled median/signature (~ms).
Eval stalls the worker loop ~1-2s once per interval (preview freezes briefly;
water_gap is delayed, never corrupted).
"""
import csv
import datetime
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml
from loguru import logger

from server.algo import AlgoResult, Annotation, Event

from night_lamp import adaptive as AD
from night_lamp import detector as DET
from night_lamp import night_state as NS
from night_lamp.evidence import Jsonl, save_snap

HERE = Path(__file__).resolve().parent
SIG_W, SIG_H = 48, 27
MAX_GAP_S = 2.0      # single arrival gap over this -> burst degraded.
# Worker loop runs ~5.5fps with water_gap (not 12.5fps stream rate), so a
# 40-frame burst nominally spans ~7.3s with jitter: budgets are sized to the
# WORKER rate, not the stream rate. Subsampled AC validity (lag window was
# calibrated @12.49fps consecutive) is tonight's measurement question:
# actual_fps is recorded per burst for morning comparison. Duty estimate is
# unbiased under uniform subsampling; AC peaks at period multiples.
MAX_SPAN_S = 15.0    # 40-frame burst spanning longer -> degraded (~2x slack)
ENV_INTERVAL = "NIGHT_LAMP_BURST_S"  # stress-test cadence override, seconds


def _sig(gray):
    return cv.resize(gray, (SIG_W, SIG_H),
                     interpolation=cv.INTER_AREA).tobytes()


def load_calibration(fp):
    """Read a night_lamp production YAML + registry CSV (server-facing subset).

    Returns calib dict; camera.* is IGNORED here (server owns rtsp_url).
    Keeps main.py's guards: frozen detector shape + registry frozen/count.
    """
    fp = Path(fp)
    cfg = yaml.safe_load(fp.read_text(encoding="utf-8"))
    base = fp.parent
    det = cfg["detector"]
    assert det["roi_r"] == 10 and det["lag_lo"] == 5, "frozen shape guard"
    night = cfg["night"]
    assert night["enter_threshold"] < night["exit_threshold"], "hysteresis required"
    reg = _load_registry(base, cfg["registry"])
    ex = cfg.get("extra_models", {}) or {}
    return {"camera_id": cfg["camera"]["id"], "registry": reg,
            "detector": det, "night": night,
            "burst_frames": int(cfg["burst"]["frames"]),
            "interval_sec": float(cfg["burst"]["interval_sec"]),
            "f2_swing_floor": float(ex.get("f2_swing_floor", AD.SWING_FLOOR)),
            "dead_bright_cap": float(ex.get("dead_bright_cap",
                                            AD.DEAD_BRIGHT_CAP)),
            "steady_swing_lo": float(ex.get("steady_swing_lo", AD.STEADY_SWING_LO)),
            "steady_roi_hi": float(ex.get("steady_roi_hi", AD.STEADY_ROI_HI)),
            "snapshot_quality": int(cfg["evidence"].get("snapshot_quality", 70)),
            "out": str(base / cfg["evidence"]["out"]),
            "detector_version": DET.DETECTOR_VERSION}


def _load_registry(base, reg_cfg):
    import csv as _csv
    rows = [r for r in _csv.DictReader(
        open(Path(base / reg_cfg["file"]), encoding="utf-8", newline=""))
        if (r.get("kind") or "").strip()]
    assert reg_cfg["frozen"] is True
    ctrls = reg_cfg.get("controls", {}) or {}
    pos, steady = set(ctrls.get("positive", [])), set(ctrls.get("steady_check", []))
    lamps = [r for r in rows
             if r["kind"] == "lamp" and (r.get("status") or "active") == "active"]
    assert len(lamps) == reg_cfg["count"], "registry count drift"
    out = []
    for r in lamps:
        ctl = "positive" if r["id"] in pos else (
            "steady_check" if r["id"] in steady else "normal")
        out.append({"id": r["id"], "x": float(r["x"]), "y": float(r["y"]),
                    "w": float(r["w"] or 0), "h": float(r["h"] or 0),
                    "origin": r["origin"], "note": r["note"], "control": ctl})
    return {"lamps": out, "version": reg_cfg["version"],
            "count": reg_cfg["count"]}


class NightLampAlgorithm:
    """per_frame on the outside, hourly-burst on the inside. See module doc."""

    name = "night_lamp"
    cadence = "per_frame"  # fed every frame; evaluation stays hourly internally

    def __init__(self, frame_shape, calib, camera_id=""):
        self._cam = camera_id or calib["camera_id"]
        self._cal = calib
        det = calib["detector"]
        self._dk = {"roi_r": det["roi_r"], "on_max_delta": det["on_max_delta"],
                    "on_mean_delta": det["on_mean_delta"],
                    "lag_lo": det["lag_lo"], "lag_hi": det["lag_hi"],
                    "ac_th": det["ac_th"], "duty_lo": det["duty_lo"],
                    "duty_hi": det["duty_hi"], "c_ac": det["c_ac"],
                    "c_min_on": det["c_min_on"]}
        self._n = calib["burst_frames"]
        out = calib["out"]
        os.makedirs(out, exist_ok=True)
        os.makedirs(os.path.join(out, "bursts"), exist_ok=True)
        self._jl = Jsonl(out, "a").open("lamp_metrics.jsonl",
                                        "global_metrics.jsonl", "errors.jsonl",
                                        "snapshots.jsonl")
        self._ns_path = os.path.join(out, "night_state.json")
        self._state, self._count = NS.load(self._ns_path)
        self._buf, self._times = [], []   # gray frames + monotonic stamps
        self._last_sig, self._last_now = None, None
        self._last_bgr = None
        self._next_due = 0.0              # first burst starts immediately
        self._burst_idx = 0
        self._dead = {}                   # lamp_id -> consecutive dead bursts
        self._annots = [Annotation("point", (l["x"], l["y"]), l["id"], "info")
                        for l in calib["registry"]["lamps"]]

    def reset(self):
        self._buf, self._times = [], []
        self._state, self._count = NS.TWILIGHT, 0
        self._dead = {}
        self._burst_idx = 0
        self._next_due = 0.0
        self._annots = [Annotation("point", (l["x"], l["y"]), l["id"], "info")
                        for l in self._cal["registry"]["lamps"]]

    def _interval(self):
        try:
            return float(os.environ.get(ENV_INTERVAL, self._cal["interval_sec"]))
        except ValueError:
            return float(self._cal["interval_sec"])

    def step(self, frame, now):
        try:
            return self._step(frame, now)
        except Exception as e:  # never break the worker loop
            logger.warning("[{}] night_lamp step failed: {}", self._cam, e)
            return AlgoResult([], list(self._annots), {})

    def _step(self, frame, now):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if now >= self._next_due and len(self._buf) < self._n:
            sig = _sig(gray)
            if sig != self._last_sig:  # drop re-fed duplicates
                # gap only applies WITHIN a burst: first frame has no predecessor
                gap_ok = (not self._buf
                          or now - self._last_now <= MAX_GAP_S)
                self._buf.append(gray)
                self._times.append(now)
                if not gap_ok:
                    self._mark_degraded("arrival_gap", now)
            self._last_sig, self._last_now = sig, now
            self._last_bgr = frame
            if len(self._buf) >= self._n:
                return self._evaluate(now)
        else:
            self._last_now = now
        return AlgoResult([], list(self._annots))

    def _mark_degraded(self, reason, now):
        self._jl.write("errors.jsonl", {"burst_idx": self._burst_idx,
                                        "burst_status": "degraded",
                                        "failure_reason": reason})
        self._jl.flush()
        self._buf, self._times = [], []
        self._next_due = now + self._interval()
        self._burst_idx += 1

    def _evaluate(self, now):
        cal, det, night = self._cal, self._cal["detector"], self._cal["night"]
        buf, times = self._buf, self._times
        self._buf, self._times = [], []
        span = times[-1] - times[0] if len(times) > 1 else 0.0
        bid = "L%04d_%s" % (self._burst_idx, datetime.datetime.now().strftime("%H%M%S"))
        events, annots = [], []  # init BEFORE try: return path must never raise
        try:
            if span > MAX_SPAN_S:
                raise RuntimeError("burst_span %.1fs over budget" % span)
            gstats = np.array([(float(g.min()), float(np.median(g[::4, ::4])),
                                float(g.max())) for g in buf])
            gmed = round(float(np.median([float(np.median(g[::4, ::4]))
                                          for g in buf])), 1)
            self._jl.write("global_metrics.jsonl",
                           {"burst_id": bid, "global_median": gmed,
                            "n_frames": len(buf),
                            "actual_fps": round(len(buf) / span, 3) if span > 0 else -1})
            state, count, _ = NS.update(self._state, self._count, gmed,
                                        night["enter_threshold"],
                                        night["exit_threshold"],
                                        night["persistence"])
            self._state, self._count = state, count
            NS.save(self._ns_path, state, count)
            lamps_out = []
            if state == NS.NIGHT:
                for l in cal["registry"]["lamps"]:
                    rr = AD.adaptive_roi(l["w"], l["h"], det["roi_r"])
                    m = DET.lamp_metrics(buf, l["x"], l["y"],
                                         **{**self._dk, "roi_r": rr})
                    f2 = AD.f2_metrics(buf, l["x"], l["y"],
                                       lag_lo=det["lag_lo"],
                                       lag_hi=det["lag_hi"], ac_th=det["ac_th"],
                                       duty_lo=det["duty_lo"],
                                       duty_hi=det["duty_hi"],
                                       swing_floor=cal["f2_swing_floor"])
                    s = AD.steady_flag(m["n_on"], m["n_valid"],
                                       f2["f2_swing"], m["roi_median"],
                                       swing_lo=cal["steady_swing_lo"],
                                       roi_hi=cal["steady_roi_hi"])
                    m.update(f2)
                    m["profile_S"] = s
                    m.update({"burst_id": bid, "lamp_id": l["id"],
                              "roi_r": rr})
                    self._jl.write("lamp_metrics.jsonl", m)
                    lamps_out.append({"lamp_id": l["id"], "x": l["x"],
                                      "y": l["y"], "A": m["profile_A"],
                                      "B": m["profile_B"],
                                      "F2": f2["profile_F2"], "S": s,
                                      "swing": f2["f2_swing"]})
                    # P0 machine (flat AND dim counts; small-but-bright
                    # flashers escape via bright cap, never accused)
                    lid = l["id"]
                    if f2["profile_F2"] == 1 or s == 1:
                        self._dead[lid] = 0
                        annots.append(Annotation("point", (l["x"], l["y"]),
                                                 lid, "info"))
                    elif (f2["f2_swing"] < cal["f2_swing_floor"]
                          and f2["f2_max"] < cal["dead_bright_cap"]):
                        self._dead[lid] = self._dead.get(lid, 0) + 1
                        n = self._dead[lid]
                        if n >= 2:
                            ev = Event(self._cam, "night_lamp", now, "alarm",
                                       {"lamp_id": lid, "x": l["x"], "y": l["y"],
                                        "dead_count": n, "swing": f2["f2_swing"],
                                        "burst_id": bid})
                            events.append(ev)
                            annots.append(Annotation("point", (l["x"], l["y"]),
                                                     lid, "alarm"))
                        else:
                            events.append(Event(
                                self._cam, "night_lamp", now, "suspected",
                                {"lamp_id": lid, "x": l["x"], "y": l["y"],
                                 "dead_count": n, "swing": f2["f2_swing"],
                                 "burst_id": bid}))
                            annots.append(Annotation("point", (l["x"], l["y"]),
                                                     lid, "suspected"))
                    else:  # borderline: silent, reset
                        self._dead[lid] = 0
                        annots.append(Annotation("point", (l["x"], l["y"]),
                                                 lid, "info"))
                snap = self._save_snap(bid, cal)
                if snap:
                    for e in events:
                        if e.kind == "alarm" and e.evidence_path is None:
                            e.evidence_path = snap
            else:
                annots = list(self._annots)
            self._annots = annots
            json.dump({"burst_id": bid, "status": "OK",
                       "global_median": gmed, "night_state": state,
                       "lamps": lamps_out},
                      open(os.path.join(cal["out"], "bursts", bid + ".json"),
                           "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:
            self._jl.write("errors.jsonl", {"burst_id": bid,
                                            "burst_status": "failed",
                                            "failure_reason": "%s:%s" % (
                                                type(e).__name__, e)})
        finally:
            self._next_due = now + self._interval()
            self._burst_idx += 1
            self._last_eval = now
            self._jl.flush()  # crash-safe: hourly evidence must hit disk
        return AlgoResult(events, list(self._annots))

    def _save_snap(self, bid, cal):
        try:
            if self._last_bgr is None:
                return None
            tag = "%s_%s" % (bid, datetime.datetime.now().strftime("%H%M%S"))
            _, ovl_rel = save_snap(cal["out"], self._last_bgr, tag,
                                   cal["registry"]["lamps"], [],
                                   cal["snapshot_quality"])
            self._jl.write("snapshots.jsonl",
                           {"kind": "burst", "overlay": ovl_rel,
                            "burst_id": bid})
            return ovl_rel
        except Exception as e:
            self._jl.write("errors.jsonl", {"burst_id": bid,
                                            "snapshot_status": "failed",
                                            "failure_reason": "%s:%s" % (
                                                type(e).__name__, e)})
            return None
