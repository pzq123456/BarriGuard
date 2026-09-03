import cv2 as cv
import numpy as np
from pathlib import Path

SRC = Path("tmp/image.png")
OUT = Path("tmp/out")
OUT.mkdir(parents=True, exist_ok=True)

img = cv.imread(str(SRC))
h, w = img.shape[:2]
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
H, S, V = cv.split(hsv)
A = lab[:, :, 1]

# 红底（种子）
red = (cv.inRange(H, 0, 10) | cv.inRange(H, 168, 180)) \
    & cv.inRange(S, 60, 255) & cv.inRange(A, 145, 255)
red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))

# 白面板（前景）：路面 V≈160 被排除，天空不与面板连通
white = cv.inRange(V, 195, 255) & cv.inRange(S, 0, 60)
white = cv.morphologyEx(white, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
cv.imwrite(str(OUT / "mask_white2.png"), white)

fg = red | white
fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# 种子 = 红底连通域（a*/V 过滤，排除锈迹/砖墙/土坡）；sure-bg = 前景膨胀区之外
n, labels, stats, _ = cv.connectedComponentsWithStats(red, 8)
markers = np.zeros((h, w), np.int32)
bg = cv.dilate(fg, np.ones((15, 15), np.uint8)) == 0
markers[bg] = 1
min_area = int(w * h * 3e-5)
seeds = 0
for i in range(1, n):
    if stats[i, cv.CC_STAT_AREA] < min_area:
        continue
    x, y, bw, bh = (stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP],
                    stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT])
    m = labels[y:y + bh, x:x + bw] == i
    if A[y:y + bh, x:x + bw][m].mean() < 155 or V[y:y + bh, x:x + bw][m].mean() < 170:
        continue
    seeds += 1
    markers[labels == i] = seeds + 1
print(f"seeds: {seeds}")

markers = markers.astype(np.int32)
cv.watershed(img, markers)

# 可视化：每只水马一个颜色 + 轮廓 + 编号
rng = np.random.default_rng(7)
palette = rng.integers(60, 255, (seeds + 2, 3), np.uint8)
palette[:2] = (40, 40, 40)
vis = palette[np.clip(markers, 0, seeds + 1)]
vis = cv.addWeighted(vis, 0.55, img, 0.45, 0)
vis[markers == -1] = (255, 255, 255)

# 找每块区域的质心标注编号
for s in range(2, seeds + 2):
    ys, xs = np.where(markers == s)
    if len(xs) < 200:
        continue
    cx, cy = int(xs.mean()), int(ys.mean())
    cv.putText(vis, str(s - 1), (cx - 8, cy), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(vis, str(s - 1), (cx - 8, cy), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 0), 1, cv.LINE_AA)

cv.imwrite(str(OUT / "barrier_watershed.png"), vis)
print(f"regions drawn, image {w}x{h}")
