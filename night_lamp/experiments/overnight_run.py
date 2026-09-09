"""Overnight stability experiment: OBSERVATION ONLY (instrumentation, no detector change).

Frozen detector (documented, not modified):
  ON rule (night_lamp/README): 7x7max >= ROI median+40 OR 7x7mean >= ROI median+30
  Profile A: n_on >= 1
  Profile B (pipeline v3档): AC(lag5..12) >= 0.35 AND 0.05 <= duty <= 0.85
  Profile C (strict OBSERVATIONAL subset): B AND n_on >= 2 AND AC >= 0.50
Registry frozen: 30 lamps, never promoted/edited here. Candidates recorded only.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

import cv2 as cv
import numpy as np

from stream_source import FileSrc, RTSPSrc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "overnight_20260908")
VID = r"C:\Users\admin\Desktop\work\BarriGuard\tmp\1749_202609040100.mp4"
RTSP_1749 = "rtsp://118.140.234.166:8554/dahua1002490"  # 镜像 server/config.yaml 1749

BURST_LEN = 40
N_BURSTS = 10
ROI_R = 10
LAG_LO, LAG_HI = 5, 12
AC_TH, DUTY_LO, DUTY_HI = 0.35, 0.05, 0.85
C_AC, C_MIN_ON = 0.50, 2
DETECTOR_VERSION = ("frozen-20260908: ON=7x7max>=med+40|7x7mean>=med+30; "
                    "A=n_on>=1; B=AC(lag5-12)>=0.35&duty[0.05,0.85]; "
                    "C=B&n_on>=2&AC>=0.50(obs)")
K7 = np.ones((7, 7), np.uint8)
K3 = np.ones((3, 3), np.uint8)
SNAP_Q = 70  # snapshot JPEG 质量（肉眼可辨即可）

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=r"C:\Users\admin\Desktop\work\BarriGuard",
                                       text=True).strip()
    except Exception:
        return "unknown"


JSONL = ("lamp_metrics.jsonl", "global_metrics.jsonl", "candidates.jsonl", "errors.jsonl",
         "snapshots.jsonl")


def full_done():
    return os.path.isfile(os.path.join(OUT, "summary.json"))


def guard(args):
    """最小数据边界保护：不删数据，只拒绝或旁移。"""
    if args.mode == "dry" and full_done():
        raise SystemExit("ERROR: summary.json 已存在（已有完整实验），dry 会截断 jsonl，拒绝执行。")
    if args.mode == "full" and full_done() and not args.resume:
        raise SystemExit("ERROR: summary.json 已存在，拒绝覆盖/追加。换目录或加 --resume 显式续跑。")
    if args.mode == "full" and not args.resume:
        for f in JSONL:  # preflight 痕迹旁移，不删除
            p = os.path.join(OUT, f)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                ts = datetime.datetime.now().strftime("%H%M%S")
                os.rename(p, p + f".preflight-{ts}.bak")
                print(f"rotate aside: {f} -> {f}.preflight-{ts}.bak", flush=True)
        bd = os.path.join(OUT, "bursts")  # preflight burst 文件移入子目录（summary 只扫顶层）
        pre = [f for f in os.listdir(bd) if f.endswith(".json")] if os.path.isdir(bd) else []
        if pre:
            ts = datetime.datetime.now().strftime("%H%M%S")
            dest = os.path.join(bd, f"preflight-{ts}")
            os.makedirs(dest, exist_ok=True)
            for f in pre:
                os.rename(os.path.join(bd, f), os.path.join(dest, f))
            print(f"rotate aside: {len(pre)} burst files -> bursts/preflight-{ts}/", flush=True)
        sd = os.path.join(OUT, "snapshots")  # preflight 快照同样旁移
        jpgs = [f for f in os.listdir(sd) if f.endswith(".jpg")] if os.path.isdir(sd) else []
        if jpgs:
            ts = datetime.datetime.now().strftime("%H%M%S")
            dest = os.path.join(sd, f"preflight-{ts}")
            os.makedirs(dest, exist_ok=True)
            for f in jpgs:
                os.rename(os.path.join(sd, f), os.path.join(dest, f))
            print(f"rotate aside: {len(jpgs)} snapshots -> snapshots/preflight-{ts}/", flush=True)


def ac_score(on):
    x = on.astype(float) - on.mean()
    if float(x @ x) <= 0:
        return 0.0, -1
    acf = np.correlate(x, x, "full")[len(x) - 1:]
    if acf[0] <= 0:
        return 0.0, -1
    acf = acf / acf[0]
    seg = acf[LAG_LO:LAG_HI + 1]
    return round(float(seg.max()), 3), int(np.argmax(seg)) + LAG_LO


def lamp_metrics(frames, x, y):
    """frames: list of gray images. Returns per-burst evidence dict."""
    H, W = frames[0].shape
    x0 = int(np.clip(round(x) - ROI_R, 0, W - 2 * ROI_R))
    y0 = int(np.clip(round(y) - ROI_R, 0, H - 2 * ROI_R))
    sig, on, meds = [], [], []
    for g in frames:
        roi = g[y0:y0 + 2 * ROI_R, x0:x0 + 2 * ROI_R]
        med = float(np.median(roi))
        mx7 = float(cv.dilate(roi, K7).max())
        mn7 = float(cv.blur(roi, (7, 7)).max())
        sig.append(mx7 - med)
        meds.append(med)
        on.append(bool(mx7 >= med + 40 or mn7 >= med + 30))
    sig = np.array(sig)
    on = np.array(on)
    n_on = int(on.sum())
    duty = round(float(on.mean()), 4)
    ac, lag = ac_score(on)
    peak = round(float(sig.max()), 1)
    pf = int(np.argmax(sig))
    pw = int((sig >= 0.5 * peak).sum()) if peak > 0 else 0
    return {"n_valid": len(frames), "n_on": n_on, "duty": duty, "ac": ac, "lag": lag,
            "min": round(float(sig.min()), 1), "median": round(float(np.median(sig)), 1),
            "max": round(float(sig.max()), 1), "p90": round(float(np.percentile(sig, 90)), 1),
            "peak": peak, "peak_frame": pf, "peak_width": pw,
            "roi_median": round(float(np.median(meds)), 1),
            "profile_A": int(n_on >= 1),
            "profile_B": int(ac >= AC_TH and DUTY_LO <= duty <= DUTY_HI),
            "profile_C": int(ac >= AC_TH and DUTY_LO <= duty <= DUTY_HI
                              and n_on >= C_MIN_ON and ac >= C_AC)}


def east_zone_candidates(frames):
    """x>1690 far-east strip mini-propose (record only)."""
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
    return [{"x": round(x, 1), "y": round(y, 1), "id": f"E{int(x)}-{int(y)}", "area": a,
             "score_range": r, "score_max": v, "source": "east_x1690_propose"}
            for x, y, a, r, v in sel]


def mid_snap(src, args, f0, starts, bi, reg, er_fh):
    """半程快照（旁路取证；取不到帧只记 failure，不影响 detector）。"""
    try:
        if args.source == "file":
            nxt = starts[bi + 1] if bi + 1 < len(starts) else f0 + BURST_LEN
            fr = src.frame_at((f0 + BURST_LEN + nxt) // 2)
        else:
            fr = src.read()
            if fr is None:
                raise RuntimeError("no_live_frame")
        wall = datetime.datetime.now().isoformat(timespec="seconds")
        tag = f"mid-B{bi:02d}_{wall[11:13]}{wall[14:16]}{wall[17:19]}"
        raw_rel, ovl_rel = save_snap(fr, tag, reg["lamps"],
                                     reg["watchlist_candidate_only_never_promote"])
        return {"kind": "mid", "raw": raw_rel, "overlay": ovl_rel, "wall_ts": wall,
                "near_burst": bi}
    except Exception as e:
        er_fh.write(json.dumps({"snapshot_status": "failed",
                                "failure_reason": f"{type(e).__name__}:{e}"}) + "\n")
        return None


def grab_burst(src, f0, n):
    """文件源确定性取帧（BGR）。直播源走 src.burst_next。"""
    return src.burst_at(f0, n)


def rep_idx(frame_meds):
    """代表帧：全局 median 最接近 burst 中位数的帧（避开闪峰/异常曝光）。"""
    meds = np.array(frame_meds)
    return int(np.argmin(np.abs(meds - np.median(meds))))


def snap_paths(tag):
    d = os.path.join(OUT, "snapshots")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, tag + ".raw.jpg"), os.path.join(d, tag + ".overlay.jpg")


def save_snap(bgr, tag, lamps, cands):
    """存 raw + overlay（纯可视化，不回流 detector）。返回 raw/overlay 相对路径。"""
    raw_p, ovl_p = snap_paths(tag)
    cv.imwrite(raw_p, bgr, [cv.IMWRITE_JPEG_QUALITY, SNAP_Q])
    vis = bgr.copy()
    for l in lamps:  # frozen registry 点+id
        col = {"P0_dead_control": (0, 0, 255), "P0_positive_control": (255, 255, 0),
               "P1_steady_but_flash": (0, 255, 255)}.get(l.get("role"), (0, 255, 0))
        cv.circle(vis, (int(l["x"]), int(l["y"])), 7, col, 2)
        cv.putText(vis, l["id"], (int(l["x"]) + 9, int(l["y"]) - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for c in cands or []:  # candidate 菱形+id（watchlist/东区）
        cv.drawMarker(vis, (int(c["x"]), int(c["y"])), (255, 0, 255),
                      cv.MARKER_DIAMOND, 16, 2)
        cv.putText(vis, c.get("id", "?"), (int(c["x"]) + 9, int(c["y"]) + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    cv.imwrite(ovl_p, vis, [cv.IMWRITE_JPEG_QUALITY, SNAP_Q])
    return os.path.relpath(raw_p, OUT), os.path.relpath(ovl_p, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "full"], default="dry")
    ap.add_argument("--source", choices=["file", "rtsp"], default="file")
    ap.add_argument("--rtsp-url", default=RTSP_1749)
    ap.add_argument("--interval-s", type=float, default=5.0,
                    help="直播源 burst 间隔秒（正式挂机 3600）")
    ap.add_argument("--resume", action="store_true", help="full 显式续跑：允许追加已有实验目录")
    args = ap.parse_args()
    guard(args)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "bursts"), exist_ok=True)
    reg = json.load(open(os.path.join(OUT, "registry_frozen.json"), encoding="utf-8"))
    assert reg["registry_frozen"] is True and reg["registry_count"] == 30
    assert len(reg["lamps"]) == 30
    wl = reg["watchlist_candidate_only_never_promote"]

    if args.source == "file":
        src = FileSrc(VID)
        fps, total = src.fps(), src.total()
        starts = [int(i * (total - BURST_LEN) / (N_BURSTS - 1)) for i in range(N_BURSTS)]
    else:
        src = RTSPSrc(args.rtsp_url).start()
        fps, total = 0.0, -1  # 直播无标称值，按 burst 实测
        starts = list(range(N_BURSTS))
    if args.mode == "dry":
        starts = starts[:1]

    meta = {"experiment": "overnight_stability_20260908", "mode": args.mode,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": args.source,
            "video": VID if args.source == "file" else args.rtsp_url,
            "fps_nominal": round(fps, 3), "total_frames": total,
            "burst_len": BURST_LEN, "burst_starts": starts,
            "registry_version": reg["registry_version"], "registry_count": 30,
            "registry_frozen": True, "detector_version": DETECTOR_VERSION,
            "threshold_profile": {"A": "n_on>=1", "B": "AC>=0.35&duty[0.05,0.85]",
                                  "C": "B&n_on>=2&AC>=0.50(obs)"},
            "git_commit": git_hash(), "resume": args.resume,
            "script_version": "20260908-snap1 (detection math unchanged)",
            "snapshot": {"cadence": "burst-linked + mid-interval (interval/2)",
                         "pick": "rep-frame by global median", "quality": SNAP_Q,
                         "note": "evidence only, never feeds detector"},
            "code_changed": False, "changed_files": []}
    json.dump(meta, open(os.path.join(OUT, "run_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lm_fh = open(os.path.join(OUT, "lamp_metrics.jsonl"),
                 "a" if args.mode == "full" else "w", encoding="utf-8")
    gb_fh = open(os.path.join(OUT, "global_metrics.jsonl"),
                 "a" if args.mode == "full" else "w", encoding="utf-8")
    cd_fh = open(os.path.join(OUT, "candidates.jsonl"),
                 "a" if args.mode == "full" else "w", encoding="utf-8")
    er_fh = open(os.path.join(OUT, "errors.jsonl"),
                 "a" if args.mode == "full" else "w", encoding="utf-8")
    sn_fh = open(os.path.join(OUT, "snapshots.jsonl"),
                 "a" if args.mode == "full" else "w", encoding="utf-8")

    ok_n = 0
    for bi, f0 in enumerate(starts):
        wall_ts = datetime.datetime.now().isoformat(timespec="seconds")
        bid = f"B{bi:02d}_f{f0:05d}" if args.source == "file" else \
            f"B{bi:02d}_{wall_ts[11:13]}{wall_ts[14:16]}{wall_ts[17:19]}"
        t0 = time.time()
        try:
            if args.source == "file":
                bgr = grab_burst(src, f0, BURST_LEN)
                f_start, f_end, burst_fps = f0, f0 + BURST_LEN - 1, round(fps, 3)
            else:
                if args.mode == "full" and bi > 0:  # 半程快照仍走旁路，不改变检测协议
                    time.sleep(args.interval_s / 2)
                    m = mid_snap(src, args, f0, starts, bi, reg, er_fh)
                    if m:
                        sn_fh.write(json.dumps(m, ensure_ascii=False) + "\n")
                    time.sleep(args.interval_s / 2)
                bgr = src.burst_next(BURST_LEN)
                el = time.time() - t0
                f_start, f_end, burst_fps = -1, -1, round(BURST_LEN / el, 3)
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
            gb_fh.write(json.dumps(gb, ensure_ascii=False) + "\n")

            lamps_out = []
            for l in reg["lamps"]:
                m = lamp_metrics(frames, l["x"], l["y"])
                m.update({"burst_id": bid, "lamp_id": l["id"], "x": l["x"], "y": l["y"],
                          "type": l["type"], "role": l["role"]})
                lamps_out.append(m)
                lm_fh.write(json.dumps(m, ensure_ascii=False) + "\n")

            cands = []
            for w in wl:
                m = lamp_metrics(frames, w["x"], w["y"])
                c = {"burst_id": bid, "id": w["id"], "x": w["x"], "y": w["y"],
                     **m, "note": w["note"]}
                cands.append(c)
                cd_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            for e in east_zone_candidates(frames):
                e["burst_id"] = bid
                cands.append(e)
                cd_fh.write(json.dumps(e, ensure_ascii=False) + "\n")

            ri = rep_idx(gb["frame_medians"])  # 代表帧（旁路取证，不参与判定）
            wall_hms = wall_ts[11:13] + wall_ts[14:16] + wall_ts[17:19]
            raw_rel, ovl_rel = save_snap(bgr[ri], f"{bid}_{wall_hms}", reg["lamps"], cands)
            snap = {"kind": "burst", "raw": raw_rel, "overlay": ovl_rel,
                    "frame_idx": ri, "wall_ts": wall_ts, "burst_id": bid}
            sn_fh.write(json.dumps(snap, ensure_ascii=False) + "\n")

            json.dump({"burst_id": bid, "status": "OK", **gb,
                       "lamps": lamps_out, "candidates": cands, "snapshot": snap,
                       "elapsed_s": round(time.time() - t0, 1),
                       "disk_free_GB": round(shutil.disk_usage(OUT).free / 1e9, 1)},
                      open(os.path.join(OUT, "bursts", bid + ".json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            ok_n += 1
            by = {l["lamp_id"]: l for l in lamps_out}
            n_cand = len(cands)
            print(f"[{bid}] burst=OK fps={burst_fps:.2f} frames={BURST_LEN} registry=30/frozen "
                  f"1306:A={by['L1306']['profile_A']}/B={by['L1306']['profile_B']} "
                  f"1326:A={by['L1326']['profile_A']}/B={by['L1326']['profile_B']} "
                  f"438:A={by['L438']['profile_A']}/B={by['L438']['profile_B']} "
                  f"ac1326={by['L1326']['ac']} d1326={by['L1326']['duty']} "
                  f"cands={n_cand} disk={shutil.disk_usage(OUT).free/1e9:.1f}GB",
                  flush=True)
        except Exception as e:
            er_fh.write(json.dumps({"burst_id": bid, "burst_status": "failed",
                                    "failure_reason": f"{type(e).__name__}:{e}"}) + "\n")
            print(f"[{bid}] burst=FAILED reason={e}", flush=True)
    if args.mode == "dry":  # dry 也走一遍 mid 路径（验证用，不睡眠；须在 close 前）
        m = mid_snap(src, args, starts[0], starts, 0, reg, er_fh)
        if m:
            sn_fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"mid snapshot OK: {m['raw']}", flush=True)
    src.close()
    for fh in (lm_fh, gb_fh, cd_fh, er_fh, sn_fh):
        fh.close()

    if args.mode == "full":
        build_summary()
    print(f"done ok={ok_n}/{len(starts)} -> {OUT}", flush=True)


def build_summary():
    lamps = {}
    n_b = 0
    bids = sorted(f for f in os.listdir(os.path.join(OUT, "bursts")) if f.endswith(".json"))
    for f in bids:
        d = json.load(open(os.path.join(OUT, "bursts", f), encoding="utf-8"))
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
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("summary written", n_b, "bursts", flush=True)


if __name__ == "__main__":
    main()
