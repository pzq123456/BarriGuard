"""Soak烧机（LED烧屏思路的工程化）：把带时间戳的真实序列喂给算法，
统计报警频率/持续性，用时间换可靠性——transient 熬不过保持期，永久缺口一直报。

核心 `run_soak` 与框架无关（只认 Algorithm 协议）；CLI 接 server 取标定。
用法：python -m water_barrier.soak --camera 1749 --glob "tmp/cap_day/cap_*.png"
       --hold 60 --out soak_hold60.json
"""
import argparse
import glob
import json
import os
import time

import cv2 as cv
import numpy as np


def parse_ts(fn):
    """cap_HHMMSS.png -> 当日秒数（跨天序列需另行处理）。"""
    b = os.path.basename(fn)[4:10]
    return int(b[0:2]) * 3600 + int(b[2:4]) * 60 + int(b[4:6])


def run_soak(build, files, resave_s=300.0, draw=None, max_shots=12):
    """喂流并统计。build(shape)->algo；files 有序；draw(vis,ep_idx) 存框图可空。

    返回摘要 dict。报警 episode 按“报警帧连续”划分；证据按上升沿+节流计数。
    """
    t00 = parse_ts(files[0])
    algo, episodes = None, []
    in_alarm, last_save, saves, alarm_n = False, -1e9, 0, 0
    in_sus, sus_start, sus_eps, sus_n = False, 0, [], 0
    statuses, step_ms, n = {}, [], 0
    for f in files:
        t = parse_ts(f) - t00
        frame = cv.imread(f)
        if frame is None:
            continue
        if algo is None:
            algo = build(frame.shape)
        t1 = time.time()
        res = algo.step(frame, float(t))
        step_ms.append((time.time() - t1) * 1000)
        n += 1
        alarming = any(e.kind == "alarm" for e in res.events)
        suspected = sum(1 for e in res.events if e.kind == "suspected")
        st = res.debug.get("frame_status", "OK")
        statuses[st] = statuses.get(st, 0) + 1
        if suspected > 0:
            sus_n += 1
            if not in_sus:
                in_sus, sus_start = True, t
        elif in_sus:
            in_sus = False
            sus_eps.append({"start": sus_start, "end": t,
                            "file": os.path.basename(f)})
        if alarming:
            alarm_n += 1
            if not in_alarm:
                in_alarm = True
                episodes.append({"start": t, "file": os.path.basename(f)})
                if draw and len(episodes) <= max_shots:
                    draw(frame, res, len(episodes))
            if t - last_save >= resave_s:
                saves += 1
                last_save = t
        else:
            in_alarm = False
    return {"n_frames": n, "span_s": parse_ts(files[-1]) - t00,
            "step_ms_mean": round(float(np.mean(step_ms)), 1) if step_ms else 0.0,
            "step_ms_p95": round(float(np.percentile(step_ms, 95)), 1) if step_ms else 0.0,
            "alarm_frames": alarm_n, "alarm_episodes": len(episodes),
            "suspected_frames": sus_n, "suspected_episodes": len(sus_eps),
            "evidence_saves": saves, "statuses": statuses,
            "first_alarm_t": episodes[0]["start"] if episodes else None,
            "episodes": episodes[:20], "sus_episodes": sus_eps[:20]}


def main():
    from server import registry, render
    from server.config import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--glob", default="tmp/cap_day/cap_*.png")
    ap.add_argument("--hold", type=float, default=None,
                    help="覆盖标定的 alarm_hold_s（烧屏时长，秒）")
    ap.add_argument("--resave", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default=None)
    args = ap.parse_args()
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)
    params = load()
    calib = next(c for c in params["cameras"] if c["id"] == args.camera)["algos"]["water_gap"]
    if args.hold is not None:
        calib = {**calib, "track": {**calib["track"], "alarm_hold_s": args.hold}}

    def draw(frame, res, idx):
        vis = render.draw_annots(frame.copy(), res.annots)
        cv.imwrite(os.path.join(args.shots, f"ep{idx:02d}.jpg"), vis)

    out = run_soak(lambda shape: registry.create("water_gap", shape, calib, args.camera),
                   sorted(glob.glob(args.glob)), args.resave,
                   draw if args.shots else None)
    out["hold"] = calib["track"]["alarm_hold_s"]
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
