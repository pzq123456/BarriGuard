"""真实帧回归：v3 gap_detector 六场景端到端验证（0944/1333 存档帧）。

预注册预期：
  S1 真实缺口(0944)  -> ALARM @10s（RER 快门 0.46 达标）
  S2 真实缺口(1333)  -> ALARM @300s 升级（缺口半在深阴影，RER 0.2x 不达标，
                        依赖升级闸门；快报漏、升级兜底，人工复核）
  S3 货车压完好slot  -> SUSPECTED @11s（快门拦截瞬态误报；注意 >300s 会升级，
                        属有界代价，由人工复核消化）
  S4 间歇遮挡(3缺1盖)-> ALARM（迟滞衰减 + RER 双门协同）
  S5 告警恢复        -> ALARM 后 25s 正常 -> NORMAL（衰减回正常态）
  S6 长时无证据升级  -> RER 持续不达标 + 缺损持续 300s -> ALARM

结果写入 lab/results/<时间戳>_gap_check_real/。
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
TICK_SEC = 1.0

# slot: (rect 标定坐标, base_coverage)。GAP base=0.80 为"修复后应有覆盖"假设值
SLOTS = {
    "GAP": ((676, 415, 766, 585), 0.80),
    "NEAR": ((300, 470, 420, 690), 0.83),
    "MID": ((560, 400, 680, 620), 0.82),
    "FAR": ((1140, 300, 1200, 390), 0.33),
}
# 路面标定 ROI（三块日晒纯路面，逐帧重取；快门最优标定）
ROAD_RECTS = ((700, 700, 1000, 860), (1040, 560, 1240, 660), (600, 850, 900, 900))


def slot_mask(shape, rect):
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = rect
    m[y0:y1, x0:x1] = 255
    return m


def road_roi_bgr(img):
    """三块路面 ROI 裁齐最小宽度后竖拼成一张 2D 图，供 patch 级拟合。"""
    crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in ROAD_RECTS]
    min_w = min(c.shape[1] for c in crops)
    return np.vstack([c[:, :min_w] for c in crops])


def make_confirm(img_ref, slot_rect):
    """RER 确认闭包：对中值帧重新分割 + 逐帧重建路色模型，避免陈旧 mask 配对。"""
    cam = cfg.scale_cam(cfg.CAMERAS[cfg.DEFAULT_CAMERA], img_ref.shape[1], img_ref.shape[0])
    sm = slot_mask(img_ref.shape, slot_rect)

    def confirm(median_bgr):
        mask = process(median_bgr, cam)
        profile = RoadColorProfile(road_roi_bgr(median_bgr))
        return calc_patch_rer_vectorized(median_bgr, sm, mask, profile)
    return confirm


def run_scenario(name, frames, slot, conditions, expect, results, mock_rer=None):
    img0 = frames[0]
    rect, base = SLOTS[slot]
    tracker = SlotTracker(slot_id=hash(name) % 1000, base_coverage=base)
    confirm = (lambda _f: mock_rer) if mock_rer is not None else make_confirm(img0, rect)
    sm = slot_mask(img0.shape, rect)
    t, rer_at_confirm = 100.0, None
    for i, frame in enumerate(frames):
        tracker.push_frame(frame)
        cond = conditions[i] if conditions is not None else \
            evaluate_slot_fast(sm, process(frame, make_cam(img0)), base)
        before = tracker._accumulated_abnormal_sec
        state = tracker.update(cond, t, confirm)
        if before < 10.0 <= tracker._accumulated_abnormal_sec:
            rer_at_confirm = confirm(tracker.get_temporal_median_frame())
        t += TICK_SEC
    ok = tracker.state == expect
    results[name] = {"state": tracker.state.name, "expect": expect.name,
                     "rer_at_confirm": rer_at_confirm, "pass": ok}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: 最终={tracker.state.name} "
          f"预期={expect.name} 确认RER={rer_at_confirm}")
    return tracker


def make_cam(img):
    return cfg.scale_cam(cfg.CAMERAS[cfg.DEFAULT_CAMERA], img.shape[1], img.shape[0])


def main():
    img944 = cv.imread(str(ROOT / "data/input/1749_0944.png"))
    img1333 = cv.imread(str(ROOT / "data/input/1749_1333.png"))
    cam = make_cam(img944)
    results = {}

    print("=== 真实帧回归（快门+升级闸门） ===")
    # S1: 真实缺口(0944) 持续 11s -> RER 快门直接报警
    conds = [evaluate_slot_fast(slot_mask(img944.shape, SLOTS["GAP"][0]),
                                process(img944, cam), SLOTS["GAP"][1])] * 11
    run_scenario("S1_缺口0944", [img944] * 11, "GAP", conds, SlotState.ALARM, results)

    # S2: 真实缺口(1333) 持续 300s -> 升级闸门兜底报警
    conds = [SlotCondition.DEFECTIVE] * 301
    run_scenario("S2_缺口1333_升级", [img1333] * 301, "GAP", conds, SlotState.ALARM, results)

    # S3: 货车压完好 NEAR slot 11s
    img_truck = img944.copy()
    img_truck[560:690, 300:420] = (200, 130, 60)
    conds = [evaluate_slot_fast(slot_mask(img944.shape, SLOTS["NEAR"][0]),
                                process(img_truck, cam), SLOTS["NEAR"][1])] * 11
    run_scenario("S3_货车压完好", [img_truck] * 11, "NEAR", conds, SlotState.SUSPECTED, results)

    # S4: 间歇遮挡 3缺+1盖 x8（条件由场景给定，模拟过车节奏）
    frames, conds = [], []
    for _ in range(8):
        frames += [img944] * 3
        conds += [SlotCondition.DEFECTIVE] * 3
        frames.append(img944)
        conds.append(SlotCondition.INTACT)
    run_scenario("S4_间歇遮挡", frames, "GAP", conds, SlotState.ALARM, results)

    # S6: RER 持续为 0（模拟最顽固遮挡物）+ 缺损持续 300s -> 升级报警
    conds = [SlotCondition.DEFECTIVE] * 301
    run_scenario("S6_升级闸门", [img944] * 301, "GAP", conds, SlotState.ALARM, results,
                 mock_rer=0.0)

    # S5: S1 告警后恢复
    tracker = run_scenario("S5_恢复", [img944] * 11, "GAP",
                           [SlotCondition.DEFECTIVE] * 11, SlotState.ALARM, results)
    for i in range(25):
        tracker.update(SlotCondition.INTACT, 111.0 + i, make_confirm(img944, SLOTS["GAP"][0]))
    results["S5_恢复"]["final_state"] = tracker.state.name
    ok5 = tracker.state == SlotState.NORMAL
    results["S5_恢复"]["pass"] = ok5
    print(f"  [{'PASS' if ok5 else 'FAIL'}] S5_恢复: 25s 正常后={tracker.state.name} 预期=NORMAL")

    n_pass = sum(1 for r in results.values() if r.get("pass"))
    print(f"\n总计 {n_pass}/{len(results)} 场景通过")

    run_dir = ROOT / "lab/results" / f"{time.strftime('%Y%m%d_%H%M%S')}_gap_check_real"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"metrics -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
