"""水马实例分割（主流程）。

思路：白面板(上)+红底座(下) 是同一水马的两个部件。
  1. 颜色掩膜：红 = HSV红区 & 高饱和 & Lab a* 高；白 = 高亮度 & 低饱和
  2. 红色连通域作 watershed 种子，先切出每个红底座
  3. 第一遍未被认领的前景里，按"横向贴近某红底座"筛白板种子，补播再 watershed
  4. 后合并：垂直堆叠 + 中心线对齐的红/白碎片拼成完整水马实例

已知局限：纯白水马（无红底座）只带 * 标记，靠邻近红底座约束，成排纯白时易漏。

输出写入本轮 run 目录：mask_*.png、vis_seg.png（分割蒙层）、instances.png（逐实例蒙太奇）。
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import crop_pad, load_input, new_run_dir, save

FRAME = "1749_0944.png"

# 红底座阈值（采样结论：水马红 a*≈150-176；土坡/锈铁 a*≈120-130）
RED_H1, RED_H2, RED_H3, RED_H4 = 0, 10, 168, 180
RED_S_MIN, RED_A_MIN = 60, 145
RED_MEAN_A_MIN, RED_MEAN_V_MIN = 155, 170
# 白面板阈值：比路面亮一截
WHITE_V_MIN, WHITE_S_MAX = 195, 60
# 过滤与合并参数
RED_AREA_RATIO = 3e-5
WHITE_AREA_MIN = 400
WHITE_ASP_MIN, WHITE_ASP_MAX = 0.2, 4.5
NEAR_DIST = 80
STACK_GAP_MIN, STACK_GAP_MAX = -20, 35
MERGE_AREA_MIN = 200
LABEL_AREA_MIN = 400


def color_masks(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    H, S, V = cv.split(hsv)
    A = lab[:, :, 1]
    red = (cv.inRange(H, RED_H1, RED_H2) | cv.inRange(H, RED_H3, RED_H4)) \
        & cv.inRange(S, RED_S_MIN, 255) & cv.inRange(A, RED_A_MIN, 255)
    red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    white = cv.inRange(V, WHITE_V_MIN, 255) & cv.inRange(S, 0, WHITE_S_MAX)
    white = cv.morphologyEx(white, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv.morphologyEx(red | white, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return red, white, fg, V, A


def red_seeds(red, V, A):
    """筛选真红底座：面积、平均 a*、平均 V 三重约束（排除锈铁/土坡/警示灯）。"""
    n, labels, stats, _ = cv.connectedComponentsWithStats(red, 8)
    min_area = int(red.shape[0] * red.shape[1] * RED_AREA_RATIO)
    ids, boxes = [], []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i, :5]
        if area < min_area:
            continue
        m = labels[y:y + bh, x:x + bw] == i
        if A[y:y + bh, x:x + bw][m].mean() < RED_MEAN_A_MIN:
            continue
        if V[y:y + bh, x:x + bw][m].mean() < RED_MEAN_V_MIN:
            continue
        ids.append(i)
        boxes.append((x, y, bw, bh))
    return ids, labels, boxes


def watershed_pass(img, fg, red_ids, red_labels):
    """第一遍：背景=1，红种子=2..k，返回已认领的 markers。"""
    markers = np.zeros(img.shape[:2], np.int32)
    markers[cv.dilate(fg, np.ones((15, 15), np.uint8)) == 0] = 1
    sid = 2
    for i in red_ids:
        markers[red_labels == i] = sid
        sid += 1
    cv.watershed(img, markers)
    return markers, sid


def white_seeds(img, markers, sid, fg, red_boxes):
    """第二遍种子：未被红种子认领的前景，须横向贴近某红底座（排除天空文字/窗框/标线）。"""
    cand = fg & ((markers == 0) | (markers == 1))
    cand = cv.morphologyEx(cand, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv.connectedComponentsWithStats(cand, 8)
    ids = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i, :5]
        if area < WHITE_AREA_MIN or not WHITE_ASP_MIN <= bw / bh <= WHITE_ASP_MAX:
            continue
        near = any(max(0, rx - (x + bw), x - (rx + rw)) < NEAR_DIST
                   and max(0, ry - (y + bh), y - (ry + rh)) < NEAR_DIST
                   for rx, ry, rw, rh in red_boxes)
        if not near:
            continue
        ids.append(i)
        markers[labels == i] = sid
        sid += 1
    cv.watershed(img, markers)
    return ids, sid


def collect_regions(markers, sid, n_red):
    regions = []
    for s in range(2, sid):
        ys, xs = np.where(markers == s)
        if len(xs) < MERGE_AREA_MIN:
            continue
        regions.append({"id": s, "kind": "R" if s <= n_red + 1 else "W",
                        "x0": int(xs.min()), "y0": int(ys.min()),
                        "x1": int(xs.max()), "y1": int(ys.max()),
                        "area": len(xs), "cx": float(xs.mean())})
    return regions


def stack_merge(regions):
    """红白碎片合并：只并"白(上)+红(下)"或同色垂直堆叠，中心线要对齐。"""
    parent = {r["id"]: r["id"] for r in regions}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(regions)):
        for b in range(a + 1, len(regions)):
            ra, rb = regions[a], regions[b]
            if ra["kind"] == "R" and rb["kind"] == "R":
                continue
            top, bot = (ra, rb) if ra["y1"] <= rb["y1"] else (rb, ra)
            gap_y = bot["y0"] - top["y1"]
            if not STACK_GAP_MIN <= gap_y <= STACK_GAP_MAX:
                continue
            if not (top["y0"] <= bot["y0"] and bot["y1"] >= top["y1"]):
                continue
            tol = max(0.5 * min(top["x1"] - top["x0"], bot["x1"] - bot["x0"]), 12)
            if abs(top["cx"] - bot["cx"]) <= tol:
                parent[find(rb["id"])] = find(ra["id"])

    groups = {}
    for r in regions:
        groups.setdefault(find(r["id"]), []).append(r)
    return list(groups.values())


def group_box(g):
    return (min(q["x0"] for q in g), min(q["y0"] for q in g),
            max(q["x1"] for q in g), max(q["y1"] for q in g))


def visualize(img, markers, groups):
    vis = img.copy()
    rng = np.random.default_rng(7)
    palette = rng.integers(60, 255, (len(groups) + 1, 3), np.uint8)
    for gi, g in enumerate(sorted(groups, key=lambda g: group_box(g)[0])):
        mask = np.isin(markers, [q["id"] for q in g])
        layer = vis.copy()
        layer[mask] = palette[gi]
        vis = cv.addWeighted(layer, 0.5, vis, 0.5, 0)
        x0, y0, x1, y1 = group_box(g)
        if (x1 - x0) * (y1 - y0) < LABEL_AREA_MIN:
            continue
        kinds = "".join(sorted(set(q["kind"] for q in g)))
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv.putText(vis, str(gi), (x0, y0 - 4), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(vis, str(gi), (x0, y0 - 4), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 1, cv.LINE_AA)
    return vis


def montage(img, groups):
    """按从左到右平铺每个实例的裁剪图，便于逐个核对是否漏检。"""
    cell_h, gap = 110, 6
    tiles = []
    for g in sorted(groups, key=lambda g: group_box(g)[0]):
        c = crop_pad(img, group_box(g))
        scale = cell_h / c.shape[0]
        tiles.append(cv.resize(c, (max(1, int(c.shape[1] * scale)), cell_h)))
    rows, row, width = [], [], 0
    for t in tiles:
        if row and width + t.shape[1] + gap > 1800:
            rows.append(row)
            row, width = [], 0
        row.append(t)
        width += t.shape[1] + gap
    if row:
        rows.append(row)
    rows = [np.hstack([np.full((cell_h, gap, 3), 255, np.uint8)] + r)
            for r in rows]
    if not rows:
        return np.full((cell_h, cell_h, 3), 255, np.uint8)
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((cell_h, width - r.shape[1], 3), 255, np.uint8)])
            for r in rows]
    return np.vstack(rows)


def main():
    img = load_input(FRAME)
    run = new_run_dir("barrier_seg")
    print(f"run: {run}")

    red, white, fg, V, A = color_masks(img)
    save(red, run, "mask_red.png")
    save(white, run, "mask_white.png")

    red_ids, red_labels, red_boxes = red_seeds(red, V, A)
    markers, sid = watershed_pass(img, fg, red_ids, red_labels)
    white_ids, sid = white_seeds(img, markers, sid, fg, red_boxes)
    print(f"seeds: red {len(red_ids)}, white {len(white_ids)}")

    regions = collect_regions(markers, sid, len(red_ids))
    groups = stack_merge(regions)
    print(f"barrier instances: {len(groups)}")
    for g in sorted(groups, key=lambda g: group_box(g)[0]):
        x0, y0, x1, y1 = group_box(g)
        kinds = "+".join(sorted(q["kind"] for q in g))
        print(f"  ({x0:>4},{y0:>4})-({x1:>4},{y1:>4}) area {sum(q['area'] for q in g):>6} [{kinds}]")

    save(visualize(img, markers, groups), run, "vis_seg.png")
    save(montage(img, groups), run, "instances.png")


if __name__ == "__main__":
    main()
