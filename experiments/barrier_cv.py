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
L, A, B = cv.split(lab)

# 红色底座：HSV 红色区间 + 饱和度 + Lab a* 三重约束
# 采样结论：水马红 a*≈150-176, S 高；土坡/路面 a*≈120-130, S 低
red_hsv = cv.inRange(H, 0, 10) | cv.inRange(H, 168, 180)
red = red_hsv & cv.inRange(S, 60, 255) & cv.inRange(A, 145, 255)

# 白色面板：高亮度低饱和（会比路面亮一截，先粗提再靠连通域面积过滤）
white = cv.inRange(V, 190, 255) & cv.inRange(S, 0, 70)

red = cv.morphologyEx(red, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
red = cv.morphologyEx(red, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
white = cv.morphologyEx(white, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

cv.imwrite(str(OUT / "mask_red.png"), red)
cv.imwrite(str(OUT / "mask_white.png"), white)

n, labels, stats, cents = cv.connectedComponentsWithStats(red, 8)
vis = img.copy()
min_area = int(w * h * 3e-5)
keep = 0
print(f"{'id':>4}{'x':>6}{'y':>6}{'w':>5}{'h':>5}{'area':>7}{'asp':>5}{'fill':>5}"
      f"{'V':>5}{'a*':>5}{'whiteUp':>8}")
rows = []
for i in range(1, n):
    x, y, bw, bh, area = stats[i]
    if area < min_area:
        continue
    m = labels[y:y + bh, x:x + bw] == i
    asp = bw / bh
    fill = area / (bw * bh)
    # 白板上方便捷：blob 上方等高带内白色像素占比
    y0 = max(0, y - bh)
    band = white[y0:y, x:x + bw]
    white_up = band.mean() / 255 if band.size else 0.0
    mv = V[y:y + bh, x:x + bw][m].mean()
    ma = A[y:y + bh, x:x + bw][m].mean()
    # 采样结论：真水马红 a*>=156 且 V>=188；锈迹/砖墙/土坡 a*<=152
    if ma < 155 or mv < 170:
        continue
    if white_up < 0.10 and area < 500:
        continue
    keep += 1
    rows.append((i, x, y, bw, bh, area, asp, fill, ma, white_up))
    print(f"{i:>4}{x:>6}{y:>6}{bw:>5}{bh:>5}{area:>7}{asp:>5.2f}{fill:>5.2f}"
          f"{mv:>5.0f}{ma:>5.0f}{white_up:>8.2f}")

for r in rows:
    i, x, y, bw, bh, area = r[:6]
    cv.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv.drawMarker(vis, (int(cents[i][0]), int(cents[i][1])), (0, 255, 255), cv.MARKER_CROSS, 12, 2)
    cv.putText(vis, str(i), (x, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv.imwrite(str(OUT / "red_components.png"), vis)
print(f"image {w}x{h}, components kept {keep} (min_area={min_area}), total {n - 1}")
