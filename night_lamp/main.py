"""Night lamp production runner -- EXTRACTED from overnight_run.py, NOT IMPROVED.

Same frozen detector (detector.py), same burst math, same evidence schema.
Only changes vs overnight_run (all wiring, no math):
  - thresholds/paths/cadence come from --config YAML (no absolute paths)
  - FileSrc imported via camera.py (lives in tools/replay.py)
  - night_state.py updated + persisted per burst (additive; detector outputs untouched)
  - no preflight rotate-aside: without --resume, jsonl streams are truncated ("w");
    with --resume they append ("a"). Refuses to run full over a finished dir
    (summary.json present) unless --resume.
  - no experiment profile-C promotion logic change: A/B/C recorded as-is.

Usage:
  python main.py --config configs/config_1749.yaml                     # dry: 1 file burst
  python main.py --config configs/config_1749.yaml --mode full          # N bursts
  python main.py --config configs/config_1750.yaml --mode full --source rtsp
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

import detector as DET
import night_state as NS
from camera import open_source
from evidence import Jsonl, rep_idx, save_snap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
K3 = np.ones((3, 3), np.uint8)

JSONL = ("lamp_metrics.jsonl", "global_metrics.jsonl", "candidates.jsonl",
         "errors.jsonl", "snapshots.jsonl")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(HERE.parent),
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_config(path):
    cfg_path = Path(path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    det = cfg["detector"]
    assert det["roi_r"] == 10 and det["lag_lo"] == 5  # frozen shape guard
    return cfg, base


def load_registry(base, reg_cfg):
    """Cold-start table: kind lamp/watchlist rows. Same dict shapes as frozen json."""
    import csv as _csv
    rows = [r for r in _csv.DictReader(
        open(Path(base / reg_cfg["file"]), encoding="utf-8", newline=""))
        if (r.get("kind") or "").strip()]
    assert reg_cfg["frozen"] is True and reg_cfg["count"] == 30
    lamps = [{"id": r["id"], "x": float(r["x"]), "y": float(r["y"]),
              "r": int(float(r["r"])), "type": r["type"], "band": r["band"],
              "role": r["role"], "provenance": r["provenance"]}
             for r in rows if r["kind"] == "lamp"]
    wl = [{"id": r["id"], "x": float(r["x"]), "y": float(r["y"]), "note": r["note"]}
          for r in rows if r["kind"] == "watchlist"]
    assert len(lamps) == 30
    return {"lamps": lamps,
            "watchlist_candidate_only_never_promote": wl,
            "registry_version": reg_cfg["version"], "registry_count": 30,
            "registry_frozen": True}


def full_done(out):
    return os.path.isfile(os.path.join(out, "summary.json"))


def east_zone_candidates(frames):
    """x>1690 far-east strip mini-propose (record only). Verbatim."""
    stack = np.stack([g[360:470, 1690:1920] for g in frames])  # 40x110x230
    mx = stack.max(axis=0).astype(np.int16)
    rg = (stack.max(axis=0).astype(np.int16) - stack.min(axis=0).astype(np.int16))
    m = ((rg >= 80) & (mx >= 150)).astype(np.uint8) * 255
    m = cv.morphologyEx(m, cv.MORPH_OPEN, K3)
    ncc, _, stats, cents = cv.connectedComponentsWithStats(m, 8)
    raw = [(float(cents[i][0]) + 1690, float(cents[i][1]) + 360,
            int(stats[i, 4]), int(rg[int(cents[i][1]), int(cents[i][0])]),
            int(mx[int(cents[i][1]), int(cents[i][0])]))
           for i in range(1, ncc) if stats[i, 4] >= 8]
    raw.sort(key=lambda t: -t[3])
    sel = []
    for x, y, a, r, v in raw:
        if all((x - sx) ** 2 + (y - sy) ** 2 >= 18 ** 2 for sx, sy, _, _, _ in sel):
            sel.append((x, y, a, r, v))
    return [{"x": round(x, 1), "y": round(y, 1), "id": "E%s-%s" % (int(x), int(y)),
             "area": a, "score_range": r, "score_max": v,
             "source": "east_x1690_propose"}
            for x, y, a, r, v in sel]


def mid_snap(jl, src, source, f0, starts, bi, reg, out, burst_len, snap_q):
    """Mid-interval snapshot (evidence side-path; failure only logged). Verbatim."""
    try:
        if source == "file":
            nxt = starts[bi + 1] if bi + 1 < len(starts) else f0 + burst_len
            fr = src.frame_at((f0 + burst_len + nxt) // 2)
        else:
            fr = src.read()
            if fr is None:
                raise RuntimeError("no_live_frame")
        wall = datetime.datetime.now().isoformat(timespec="seconds")
        tag = "mid-B%02d_%s%s%s" % (bi, wall[11:13], wall[14:16], wall[17:19])
        raw_rel, ovl_rel = save_snap(out, fr, tag, reg["lamps"],
                                     reg["watchlist_candidate_only_never_promote"],
                                     snap_q)
        return {"kind": "mid", "raw": raw_rel, "overlay": ovl_rel, "wall_ts": wall,
                "near_burst": bi}
    except Exception as e:
        jl.write("errors.jsonl", {"snapshot_status": "failed",
                                  "failure_reason": "%s:%s" % (type(e).__name__, e)})
        return None


def det_kwargs(det):
    return {"roi_r": det["roi_r"], "on_max_delta": det["on_max_delta"],
            "on_mean_delta": det["on_mean_delta"], "lag_lo": det["lag_lo"],
            "lag_hi": det["lag_hi"], "ac_th": det["ac_th"],
            "duty_lo": det["duty_lo"], "duty_hi": det["duty_hi"],
            "c_ac": det["c_ac"], "c_min_on": det["c_min_on"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_1749.yaml")
    ap.add_argument("--mode", choices=["dry", "full"], default="dry")
    ap.add_argument("--source", choices=["file", "rtsp"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-bursts", type=int, default=None)
    ap.add_argument("--interval-s", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg, base = load_config(args.config)
    det = cfg["detector"]
    night = cfg["night"]
    assert night["enter_threshold"] < night["exit_threshold"], "hysteresis required"

    source = args.source or cfg["source"]["default"]
    out = args.out or str(base / cfg["evidence"]["out"])
    n_bursts = args.num_bursts or cfg["burst"]["count"]
    burst_len = cfg["burst"]["frames"]
    interval_s = args.interval_s if args.interval_s is not None else cfg["burst"]["interval_sec"]
    snap_q = cfg["evidence"]["snapshot_quality"]

    if args.mode == "dry" and full_done(out):
        raise SystemExit("ERROR: summary.json exists, dry would truncate jsonl. Refusing.")
    if args.mode == "full" and full_done(out) and not args.resume:
        raise SystemExit("ERROR: summary.json exists. Use --resume or a fresh --out.")

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "bursts"), exist_ok=True)
    reg = load_registry(base, cfg["registry"])
    wl = reg["watchlist_candidate_only_never_promote"]
    dk = det_kwargs(det)

    ns_path = os.path.join(out, "night_state.json")
    if not args.resume and os.path.isfile(ns_path):
        os.remove(ns_path)
    state, count = NS.load(ns_path) if args.resume else (NS.TWILIGHT, 0)

    if source == "file":
        video = str(base / cfg["camera"]["video"])
        src = open_source("file", video_path=video)
        fps, total = src.fps(), src.total()
        starts = [int(i * (total - burst_len) / (n_bursts - 1)) for i in range(n_bursts)]
    else:
        src = open_source("rtsp", rtsp_url=cfg["camera"]["rtsp"])
        fps, total = 0.0, -1
        starts = list(range(n_bursts))
    if args.mode == "dry":
        starts = starts[:1]

    meta = {"experiment": "night_lamp_production", "mode": args.mode,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": os.path.basename(args.config),
            "camera_id": cfg["camera"]["id"],
            "source": source,
            "video": str(base / cfg["camera"]["video"]) if source == "file" else cfg["camera"]["rtsp"],
            "fps_nominal": round(fps, 3), "total_frames": total,
            "burst_len": burst_len, "burst_starts": starts,
            "registry_version": reg["registry_version"], "registry_count": 30,
            "registry_frozen": True, "detector_version": DET.DETECTOR_VERSION,
            "threshold_profile": {"A": "n_on>=1", "B": "AC>=0.35&duty[0.05,0.85]",
                                  "C": "B&n_on>=2&AC>=0.50(obs)"},
            "night": {"version": NS.NIGHT_VERSION, "enter_threshold": night["enter_threshold"],
                      "exit_threshold": night["exit_threshold"], "persistence": night["persistence"],
                      "require_night_for_detection": night["require_night_for_detection"],
                      "note": "100/120/p=2 is provisional, derived from one night of "
                              "qualification data; production starting point, not a "
                              "validated universal threshold"},
            "git_commit": git_hash(), "resume": args.resume,
            "script_version": "production-p1 (extracted from overnight 20260908-snap1, math unchanged)",
            "snapshot": {"cadence": "burst-linked + mid-interval (interval/2)",
                         "pick": "rep-frame by global median", "quality": snap_q,
                         "note": "evidence only, never feeds detector"}}
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    jl = Jsonl(out, "a" if (args.mode == "full" and args.resume) else "w").open(*JSONL)

    ok_n = 0
    for bi, f0 in enumerate(starts):
        wall_ts = datetime.datetime.now().isoformat(timespec="seconds")
        bid = ("B%02d_f%05d" % (bi, f0)) if source == "file" else \
            ("B%02d_%s%s%s" % (bi, wall_ts[11:13], wall_ts[14:16], wall_ts[17:19]))
        t0 = time.time()
        try:
            if source == "file":
                bgr = src.burst_at(f0, burst_len)
                f_start, f_end, burst_fps = f0, f0 + burst_len - 1, round(fps, 3)
            else:
                if args.mode == "full" and bi > 0:
                    time.sleep(interval_s / 2)
                    m = mid_snap(jl, src, source, f0, starts, bi, reg, out, burst_len, snap_q)
                    if m:
                        jl.write("snapshots.jsonl", m)
                    time.sleep(interval_s / 2)
                bgr = src.burst_next(burst_len)
                el = time.time() - t0
                f_start, f_end, burst_fps = -1, -1, round(burst_len / el, 3)
            frames = [cv.cvtColor(fr, cv.COLOR_BGR2GRAY) for fr in bgr]
            gstats = []
            for g in frames:
                s = g[::4, ::4]
                gstats.append((float(s.min()), float(np.median(s)),
                               float(np.percentile(s, 90)), float(s.max())))
            gstats = np.array(gstats)
            gb = {"burst_id": bid, "frame_start": f_start, "frame_end": f_end,
                  "n_frames": len(frames), "actual_fps": burst_fps,
                  "global_min": round(float(gstats[:, 0].min()), 1),
                  "global_median": round(float(np.median(gstats[:, 1])), 1),
                  "global_p90": round(float(np.median(gstats[:, 2])), 1),
                  "global_max": round(float(gstats[:, 3].max()), 1),
                  "frame_medians": [round(float(v), 1) for v in gstats[:, 1]]}
            jl.write("global_metrics.jsonl", gb)

            # Night qualification: global_median only, persisted cross-burst.
            # Product gate (no tuning): non-NIGHT bursts skip detector entirely.
            state, count, entered = NS.update(state, count, gb["global_median"],
                                              night["enter_threshold"], night["exit_threshold"],
                                              night["persistence"])
            NS.save(ns_path, state, count)
            gated = bool(night["require_night_for_detection"] and state != NS.NIGHT)
            ns_rec = {"state": state, "entered": entered, "detection_gated": gated,
                      "enter_threshold": night["enter_threshold"],
                      "exit_threshold": night["exit_threshold"]}

            lamps_out, cands = [], []
            if not gated:
                for l in reg["lamps"]:
                    m = DET.lamp_metrics(frames, l["x"], l["y"], **dk)
                    m.update({"burst_id": bid, "lamp_id": l["id"], "x": l["x"], "y": l["y"],
                              "type": l["type"], "role": l["role"]})
                    lamps_out.append(m)
                    jl.write("lamp_metrics.jsonl", m)

                for w in wl:
                    m = DET.lamp_metrics(frames, w["x"], w["y"], **dk)
                    c = {"burst_id": bid, "id": w["id"], "x": w["x"], "y": w["y"],
                         **m, "note": w["note"]}
                    cands.append(c)
                    jl.write("candidates.jsonl", c)
                for e in east_zone_candidates(frames):
                    e["burst_id"] = bid
                    cands.append(e)
                    jl.write("candidates.jsonl", e)

            ri = rep_idx(gb["frame_medians"])
            wall_hms = wall_ts[11:13] + wall_ts[14:16] + wall_ts[17:19]
            raw_rel, ovl_rel = save_snap(out, bgr[ri], "%s_%s" % (bid, wall_hms),
                                         reg["lamps"], cands, snap_q)
            snap = {"kind": "burst", "raw": raw_rel, "overlay": ovl_rel,
                    "frame_idx": ri, "wall_ts": wall_ts, "burst_id": bid}
            jl.write("snapshots.jsonl", snap)

            json.dump({"burst_id": bid, "status": "OK", **gb,
                       "night_state": ns_rec,
                       "lamps": lamps_out, "candidates": cands, "snapshot": snap,
                       "elapsed_s": round(time.time() - t0, 1),
                       "disk_free_GB": round(shutil.disk_usage(out).free / 1e9, 1)},
                      open(os.path.join(out, "bursts", bid + ".json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            ok_n += 1
            if gated:
                print("[%s] burst=OK fps=%.2f frames=%s registry=30/frozen night=%s GATED "
                      "(G=%s, no detection)" % (bid, burst_fps, burst_len, state, gb["global_median"]),
                      flush=True)
            else:
                by = {l["lamp_id"]: l for l in lamps_out}
                print("[%s] burst=OK fps=%.2f frames=%s registry=30/frozen night=%s "
                      "1306:A=%s/B=%s 1326:A=%s/B=%s 438:A=%s/B=%s "
                      "ac1326=%s d1326=%s cands=%s" %
                      (bid, burst_fps, burst_len, state,
                       by["L1306"]["profile_A"], by["L1306"]["profile_B"],
                       by["L1326"]["profile_A"], by["L1326"]["profile_B"],
                       by["L438"]["profile_A"], by["L438"]["profile_B"],
                       by["L1326"]["ac"], by["L1326"]["duty"], len(cands)), flush=True)
        except Exception as e:
            jl.write("errors.jsonl", {"burst_id": bid, "burst_status": "failed",
                                      "failure_reason": "%s:%s" % (type(e).__name__, e)})
            print("[%s] burst=FAILED reason=%s" % (bid, e), flush=True)
    if args.mode == "dry":
        m = mid_snap(jl, src, source, starts[0], starts, 0, reg, out, burst_len, snap_q)
        if m:
            jl.write("snapshots.jsonl", m)
            print("mid snapshot OK: %s" % m["raw"], flush=True)
    src.close()
    jl.close()

    if args.mode == "full":
        build_summary(out)
    print("done ok=%s/%s -> %s" % (ok_n, len(starts), out), flush=True)


def build_summary(out):
    lamps = {}
    n_b = 0
    bids = sorted(f for f in os.listdir(os.path.join(out, "bursts")) if f.endswith(".json"))
    for f in bids:
        d = json.load(open(os.path.join(out, "bursts", f), encoding="utf-8"))
        if d.get("status") != "OK":
            continue
        n_b += 1
        for l in d["lamps"]:
            s = lamps.setdefault(l["lamp_id"], {"x": l["x"], "y": l["y"], "type": l["type"],
                                                "role": l["role"], "A": [], "B": [], "C": [],
                                                "ac": [], "duty": [], "n_on": []})
            s["A"].append(l["profile_A"])
            s["B"].append(l["profile_B"])
            s["C"].append(l["profile_C"])
            s["ac"].append(l["ac"])
            s["duty"].append(l["duty"])
            s["n_on"].append(l["n_on"])
    per = {}
    for lid, s in lamps.items():
        a = np.array(s["ac"])
        du = np.array(s["duty"])
        per[lid] = {"x": s["x"], "y": s["y"], "type": s["type"], "role": s["role"],
                    "n": len(s["A"]), "A_rate": round(float(np.mean(s["A"])), 3),
                    "B_rate": round(float(np.mean(s["B"])), 3),
                    "C_rate": round(float(np.mean(s["C"])), 3),
                    "mean_ac": round(float(a.mean()), 3), "std_ac": round(float(a.std()), 3),
                    "mean_duty": round(float(du.mean()), 4), "std_duty": round(float(du.std()), 4),
                    "mean_n_on": round(float(np.mean(s["n_on"])), 1)}
    json.dump({"total_bursts": n_b, "registry_count": 30, "registry_frozen": True,
               "P0": {k: per[k] for k in ("L1306", "L1326") if k in per},
               "P1": {k: per[k] for k in ("L438",) if k in per},
               "per_lamp": per},
              open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("summary written", n_b, "bursts", flush=True)


if __name__ == "__main__":
    main()
