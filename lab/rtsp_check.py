"""RTSP 实时流端到端验证：拉流 -> 分割 -> slot 快检 -> 状态机(含 RER 确认)。

预注册预期（缺口仍在现场）：
  GAP  slot -> DEFECTIVE -> ALARM @~10s（RER 确认）
  NEAR/MID slot -> INTACT（车流不落在水马带上）
  FAR  slot -> 允许过车瞬间抖动，迟滞衰减应吸收
结果写入 lab/results/<时间戳>_rtsp_check/。
"""
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.gap_detector import (  # noqa: E402
    RoadColorProfile, SlotCondition, SlotState, SlotTracker,
    calc_patch_rer_vectorized, evaluate_slot_fast,
)
from server import config as cfg  # noqa: E402
from server.barrier import process  # noqa: E402
from server.source import Reader  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RUN_SEC = 90.0

# 标定坐标(1622x905)下的 slot 与路面标定 ROI，按实流分辨率等比缩放
SLOTS_CALIB = {
    "GAP": ((676, 415, 766, 585), 0.80),
    "NEAR": ((300, 470, 420, 690), 0.83),
    "MID": ((560, 400, 680, 620), 0.82),
    "FAR": ((1140, 300, 1200, 390), 0.33),
}
ROAD_RECTS_CALIB = ((700, 700, 1000, 860), (1040, 560, 1240, 660), (600, 850, 900, 900))
STATE_COLOR = {SlotState.NORMAL: (120, 200, 120), SlotState.SUSPECTED: (80, 160, 255),
               SlotState.ALARM: (80, 80, 255)}


def scale_rect(rect, w, h):
    sx, sy = w / cfg.CALIB_W, h / cfg.CALIB_H
    x0, y0, x1, y1 = rect
    return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))


def main():
    reader = Reader(cfg.RTSP_1749)
    reader.start()

    frame = None
    t_deadline_open = time.monotonic() + 20.0
    while frame is None and time.monotonic() < t_deadline_open:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
    if frame is None:
        sys.exit("20s 内未取到首帧")
    h, w = frame.shape[:2]
    print(f"流已连接 {w}x{h}")

    slots, trackers, stats = {}, {}, {}
    for idx, (name, (rect, base)) in enumerate(SLOTS_CALIB.items()):
        srect = scale_rect(rect, w, h)
        sm = np.zeros((h, w), np.uint8)
        x0, y0, x1, y1 = srect
        sm[y0:y1, x0:x1] = 255
        slots[name] = sm
        trackers[name] = SlotTracker(slot_id=idx, base_coverage=base)
        stats[name] = {"cov": [], "transitions": 0, "last_state": SlotState.NORMAL,
                       "rer": [], "rect": srect}

    def make_confirm(name):
        sm = slots[name]

        def confirm(median_bgr):
            mask = process(median_bgr, cam)
            crops = [median_bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in road_rects]
            min_w = min(c.shape[1] for c in crops)
            profile = RoadColorProfile(np.vstack([c[:, :min_w] for c in crops]))
            return calc_patch_rer_vectorized(median_bgr, sm, mask, profile)
        return confirm

    road_rects = [scale_rect(r, w, h) for r in ROAD_RECTS_CALIB]
    cam = cfg.scale_cam(cfg.CAMERAS[cfg.DEFAULT_CAMERA], w, h)
    confirms = {name: make_confirm(name) for name in slots}

    t0 = time.monotonic()
    t_next_log = t0 + 10.0
    n_frames, last_frame = 0, None
    print(f"运行 {RUN_SEC:.0f}s ...")
    while time.monotonic() - t0 < RUN_SEC:
        f = reader.read()
        if f is None:
            time.sleep(0.02)
            continue
        now = time.monotonic()
        last_frame = f
        mask = process(f, cam)
        n_frames += 1
        for name, sm in slots.items():
            tr = trackers[name]
            cond = evaluate_slot_fast(sm, mask, tr.base_coverage)
            cov = cv.countNonZero(cv.bitwise_and(sm, mask)) / cv.countNonZero(sm)
            stats[name]["cov"].append(cov)
            st = tr.update(cond, now, confirms[name])
            tr.push_frame(f)
            if st is not stats[name]["last_state"]:
                stats[name]["transitions"] += 1
                print(f"  [{now - t0:6.1f}s] {name}: {stats[name]['last_state'].name} -> {st.name}")
                stats[name]["last_state"] = st
        if now >= t_next_log:
            covs = ", ".join(f"{n}={np.mean(s['cov']):.2f}" for n, s in stats.items())
            print(f"  [{now - t0:6.1f}s] fps={n_frames / (now - t0):.1f} 覆盖率均值 {covs}")
            t_next_log += 10.0

    dur = time.monotonic() - t0
    print(f"\n=== 结果 ({n_frames} 帧 / {dur:.0f}s, {n_frames / dur:.1f} fps) ===")
    metrics = {"resolution": f"{w}x{h}", "duration_sec": round(dur, 1),
               "frames": n_frames, "fps": round(n_frames / dur, 2), "slots": {}}
    for name, s in stats.items():
        covs = s["cov"]
        metrics["slots"][name] = {
            "final_state": trackers[name].state.name,
            "coverage_min_mean_max": [round(min(covs), 2), round(float(np.mean(covs)), 2),
                                      round(max(covs), 2)],
            "transitions": s["transitions"],
            "rect_scaled": s["rect"],
        }
        print(f"  {name}: 最终={trackers[name].state.name} "
              f"覆盖率 min/mean/max={min(covs):.2f}/{np.mean(covs):.2f}/{max(covs):.2f} "
              f"跳变={s['transitions']}")

    if last_frame is not None:
        vis = last_frame.copy()
        mask = process(last_frame, cam)
        layer = vis.copy()
        layer[mask > 0] = (80, 220, 80)
        vis = cv.addWeighted(layer, 0.45, vis, 0.55, 0)
        for name, s in stats.items():
            x0, y0, x1, y1 = s["rect"]
            color = STATE_COLOR[trackers[name].state]
            cv.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv.putText(vis, f"{name}:{trackers[name].state.name}", (x0, y0 - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        run_dir = ROOT / "lab/results" / f"{time.strftime('%Y%m%d_%H%M%S')}_rtsp_check"
        run_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(run_dir / "final_overlay.png"), vis)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print(f"输出 -> {run_dir}")


if __name__ == "__main__":
    main()
