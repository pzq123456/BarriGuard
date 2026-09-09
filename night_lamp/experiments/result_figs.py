import json
import sys
from pathlib import Path

import cv2 as cv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).parent
VID = Path(r"C:\Users\admin\Desktop\work\BarriGuard\tmp\1749_202609040100.mp4")
r = json.load(open(OUT / "pipeline_result.json"))
cap = cv.VideoCapture(str(VID))
cap.set(cv.CAP_PROP_POS_FRAMES, 300)
ok, ref = cap.read()
cap.release()
if not ok:
    raise SystemExit("cannot read base frame from video")


def label(img, x, y, txt, color):
    cv.circle(img, (int(x), int(y)), 7, color, 2)
    cv.putText(img, txt, (int(x) - 24, int(y) - 12), cv.FONT_HERSHEY_SIMPLEX,
               0.45, color, 1)


vis = ref.copy()
for a in r["alive_flash"]:
    label(vis, a["x"], a["y"], "F", (0, 255, 0))
for s in r["alive_steady"]:
    label(vis, s["x"], s["y"], "S", (255, 255, 0))
for d in r["dead"]:
    cv.rectangle(vis, (int(d["x"]) - 14, int(d["y"]) - 14),
                 (int(d["x"]) + 14, int(d["y"]) + 14), (0, 0, 255), 2)
    cv.putText(vis, "DEAD", (int(d["x"]) - 26, int(d["y"]) + 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
cv.imwrite(str(OUT / "pipeline_map.png"), vis)

# far-line zoom with dead lamp annotated
crop = vis[340:480, 1150:1920]
crop = cv.resize(crop, None, fx=2.2, fy=2.2, interpolation=cv.INTER_CUBIC)
cv.imwrite(str(OUT / "result_far_zoom.png"), crop)

# dead lamp close-up: dead + two alive neighbours
crop = vis[380:450, 1230:1380]
crop = cv.resize(crop, None, fx=5.0, fy=5.0, interpolation=cv.INTER_CUBIC)
cv.imwrite(str(OUT / "result_dead_closeup.png"), crop)

# right band verification crop (periodic spots on right line?)
crop = ref[600:900, 1550:1750]
crop = cv.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv.INTER_CUBIC)
cv.imwrite(str(OUT / "result_right_band.png"), crop)
print("saved: pipeline_map.png, result_far_zoom.png, result_dead_closeup.png, result_right_band.png")

