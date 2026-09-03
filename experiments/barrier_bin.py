"""水马语义二值化：直接输出 0/255 屏障掩膜，不拆实例（新路线试验）。

假设：放弃 watershed/seed（为拆实例而生，Seed 面积过滤正是远端漏检根因），改用
  "红∪白粗提取 × 近严远宽双阈值 × ROI 几何约束 × 定向闭运算连成屏障带 × 连通域清杂"，
  直接产出业务所需的二值 mask，且远端 15~30px 小水马不再成段漏检。

预注册验收标准（跑完逐条核对，不许事后改口）：
  A1 (自动) mask_barrier.png：单通道、同尺寸、像素值 ⊆ {0,255}
  A2 (自动) 高置信参考召回 ≥97%：参考 = 严格红CC≥30px ∪ 严格白CC（远端带内≥20px，带外≥300px），限 ROI
  A3 (自动) 桥头远端带内每个参考 CC 覆盖 ≥80%（远端不再整段漏检）
  B1 (看图) 近/中景左排水马带完整，无成块漏检
  B2 (看图) 桥头两侧远端水马带连续，断缝 ≤2 处
  B3 (看图) 天空/树/绿网/路面零误检
  B4 (看图) 右下土坡、锈铁、孤立警示灯不误检

输出 run 目录：mask_barrier.png（最终交付物）、vis_bin.png（ROI+掩膜叠加）、
chk_miss.png（参考漏检点，红点标出）。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_input, new_run_dir, save

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRAME = "1749_0944.png"

# 严格色阈值（近/中景，与 barrier_seg 同标定：水马红 a*≈150-176；土坡/锈铁 a*≈120-130）
RED_H1, RED_H2, RED_H3, RED_H4 = 0, 10, 168, 180
RED_S_MIN, RED_A_MIN = 60, 145
WHITE_V_MIN, WHITE_S_MAX = 195, 60
# 远端带内放宽阈值（远端粉底座 a*≈126-146、S 低、色相不稳；白板实测严格阈值即可抓到）
FAR_RED_S_MIN, FAR_RED_A_MIN, FAR_RED_V_MIN = 25, 134, 140
FAR_WHITE_V_MIN, FAR_WHITE_S_MAX = WHITE_V_MIN, WHITE_S_MAX

# ROI 矩形（相机 1749 几何）：左排 / 桥头远端带 / 右排，只此相机有效
ROI_RECTS = [(0, 330, 900, 930), (1040, 295, 1462, 395), (880, 370, 1468, 950)]
# 远端放宽阈值生效带：主带 + 左远列（粉底座低洼段），主带下缘避开路面土痕(y≥395)
FAR_BAND_RECTS = [(1040, 295, 1462, 395), (1040, 375, 1170, 445)]
# 烧录物/车误检排除框：时间戳、"1749"章、桥 头白色面包车、橙色吊车驾驶室
EXCLUDE_RECTS = [(1100, 25, 1595, 135), (15, 825, 125, 900),
                 (1272, 280, 1318, 306), (1420, 272, 1458, 306)]

# 形态学：open 去斑点；close 补缝；(1,21)/(21,1) 线段核沿路向把屏障带抹平连片
OPEN_K, CLOSE_K, KNIT_H, KNIT_V = (3, 3), (9, 9), (1, 21), (21, 1)
MIN_CC_AREA, FAR_MIN_CC_AREA = 12, 6
# CC 均值色体检（防灰色路面/树影混入）：偏红 或 偏白 二选一
CC_A_REDISH, CC_V_WHITEISH, CC_S_WHITEISH = 136, 160, 80
# 参考域阈值
REF_RED_AREA, REF_WHITE_AREA, REF_WHITE_AREA_FAR = 30, 300, 20

A2_TARGET, A3_TARGET = 0.97, 0.80


def rect_img(shape, rects, val=255):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = val
    return m


def color_masks(hsv, lab, s_red, a_red, v_red, v_white, s_white):
    """一组(红,白)掩膜；远端版多一个 V 下限防暗色噪声。"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = cv.inRange(h, RED_H1, RED_H2) | cv.inRange(h, RED_H3, RED_H4)
    red = hue & cv.inRange(s, s_red, 255) & cv.inRange(lab[..., 1], a_red, 255)
    if v_red:
        red &= cv.inRange(v, v_red, 255)
    white = cv.inRange(v, v_white, 255) & cv.inRange(s, 0, s_white)
    return red, white


def build_fg(hsv, lab, far_band):
    """近严远宽粗提取 + 定向闭运算，把红底座和白板抹平成连续屏障带。"""
    red, white = color_masks(hsv, lab, RED_S_MIN, RED_A_MIN, 0, WHITE_V_MIN, WHITE_S_MAX)
    red_f, white_f = color_masks(hsv, lab, FAR_RED_S_MIN, FAR_RED_A_MIN,
                                 FAR_RED_V_MIN, FAR_WHITE_V_MIN, FAR_WHITE_S_MAX)
    for m in (red, white, red_f, white_f):
        m[:] = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones(OPEN_K, np.uint8))
    fg = (red | white) | ((red_f | white_f) & far_band)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(CLOSE_K, np.uint8))
    # 线段核沿路向收编断缝：横向成排 + 纵向红白堆叠
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(KNIT_H, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(KNIT_V, np.uint8))
    return fg


def cc_mean(hsv, lab, labels, i, x, y, w, h):
    m = labels[y:y + h, x:x + w] == i
    return (lab[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 2][y:y + h, x:x + w][m].mean())


def clean(fg, hsv, lab, far_band):
    """连通域清杂：小斑剔除；均值色既不偏红也不偏白的（灰路面/树影）剔除。"""
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        x, y, w, h, area = stats[i, :5]
        min_area = FAR_MIN_CC_AREA if far_band[y + h // 2, x + w // 2] else MIN_CC_AREA
        if area < min_area:
            continue
        a, s, v = cc_mean(hsv, lab, labels, i, x, y, w, h)
        if a >= CC_A_REDISH or (v >= CC_V_WHITEISH and s <= CC_S_WHITEISH):
            out[labels == i] = 255
    return out


def build_reference(hsv, lab, roi, far_band):
    """高置信参考：严格阈值 CC（红≥30px；白 远端带内≥20px / 带外≥300px）。"""
    red, white = color_masks(hsv, lab, RED_S_MIN, RED_A_MIN, 0, WHITE_V_MIN, WHITE_S_MAX)
    red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones(OPEN_K, np.uint8))
    white = cv.morphologyEx(white, cv.MORPH_OPEN, np.ones(OPEN_K, np.uint8))
    ref = np.zeros_like(roi)
    ccs = []
    for src, near, far in ((red, REF_RED_AREA, REF_RED_AREA), (white, REF_WHITE_AREA, REF_WHITE_AREA_FAR)):
        n, labels, stats, _ = cv.connectedComponentsWithStats(src & roi, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i, :5]
            cx, cy = x + w // 2, y + h // 2
            if area < (far if far_band[cy, cx] else near):
                continue
            cm = labels == i
            ref[cm] = 255
            ccs.append((cm, bool(far_band[cy, cx]), (x, y, x + w, y + h)))
    return ref, ccs


def overlay(img, mask, roi, excludes):
    vis = img.copy()
    layer = vis.copy()
    layer[mask > 0] = (80, 220, 80)
    vis = cv.addWeighted(layer, 0.45, vis, 0.55, 0)
    for x0, y0, x1, y1 in ROI_RECTS:
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
    for x0, y0, x1, y1 in EXCLUDE_RECTS:
        cv.rectangle(vis, (x0, y0), (x1, y1), (255, 80, 80), 1)
    return vis


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")


def main():
    frame = sys.argv[1] if len(sys.argv) > 1 else FRAME
    img = load_input(frame)
    run = new_run_dir("barrier_bin")
    print(f"frame: {frame}  run: {run}")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    roi = rect_img(img.shape, ROI_RECTS) & ~rect_img(img.shape, EXCLUDE_RECTS)
    far_band = rect_img(img.shape, FAR_BAND_RECTS) & roi

    fg = build_fg(hsv, lab, far_band)
    mask = clean(fg & roi, hsv, lab, far_band)
    save(mask, run, "mask_barrier.png")
    save(overlay(img, mask, roi, EXCLUDE_RECTS), run, "vis_bin.png")

    ref, ccs = build_reference(hsv, lab, roi, far_band)
    miss = ref & (mask == 0)
    save(cv.dilate(miss, np.ones((3, 3), np.uint8)), run, "chk_miss.png")

    print("验收（预注册标准）：")
    report("A1 二值/单通道/同尺寸",
           mask.ndim == 2 and mask.shape == img.shape[:2] and np.isin(mask, (0, 255)).all())
    cov = np.count_nonzero(mask & ref) / max(1, np.count_nonzero(ref))
    report(f"A2 参考召回 {cov:.3f} >= {A2_TARGET}", cov >= A2_TARGET)
    bad = [(bb, np.count_nonzero(mask & cm) / np.count_nonzero(cm))
           for cm, in_band, bb in ccs if in_band
           and np.count_nonzero(mask & cm) / np.count_nonzero(cm) < A3_TARGET]
    report(f"A3 远端带逐CC覆盖 (共{sum(1 for _, b, _ in ccs if b)}个)", not bad,
           "; ".join(f"{bb}:{c:.2f}" for bb, c in bad) if bad else "")
    print(f"mask 像素 {np.count_nonzero(mask)}, 参考像素 {np.count_nonzero(ref)}")


if __name__ == "__main__":
    main()
