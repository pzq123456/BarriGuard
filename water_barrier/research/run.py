"""实验入口：对给定帧跑 分割 + 自主缺口检测，输出缺口标注截图。

用法：
    python research/run.py <img1> <img2> ... [-o 输出目录] [--name 标签...]
无参数时使用内置默认帧（data/input 白天帧 + research/out 的实流帧）。

输出（每帧）：<输出目录>/<name>_annotated.png
    绿 = 分割出的水马掩膜；红框 = 算法检出的缺口（含 inter/intra 类型标注）。
"""
import argparse
import os
import sys

import cv2 as cv
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from research.segment import segment  # noqa: E402
from research.detect import detect_gaps  # noqa: E402

DATA = os.path.join(ROOT, "..", "data")
RTSP = "rtsp://118.140.234.166:8554/dahua1002490"


def fetch_live_frame(path, warm_std=35.0, warm_max=300):
    """从 RTSP 流抓一帧有效帧（H.265 解码需预热，灰度方差<warm_std 判无效）。"""
    import time
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv.VideoCapture(RTSP, cv.CAP_FFMPEG)
    cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
    if not cap.isOpened():
        print("无法打开 RTSP 流:", RTSP)
        return False
    for i in range(warm_max):
        ok, f = cap.read()
        if not ok or f is None:
            time.sleep(0.2)
            continue
        if float(cv.cvtColor(f, cv.COLOR_BGR2GRAY).std()) > warm_std:
            cv.imwrite(path, f)
            cap.release()
            print(f"已抓取实时帧（预热第{i}帧）: {path}")
            return True
    cap.release()
    return False


def annotate(frame, name, outdir):
    h, w = frame.shape[:2]
    fg = segment(frame)
    gaps = detect_gaps(fg,
                       min_area=max(2000, int(0.0008 * h * w)),
                       min_gap_px=max(20.0, 0.02 * w),
                       max_gap_px=max(60.0, 0.06 * w))
    vis = frame.copy()
    vis[fg > 0] = (60, 220, 60)
    for g in gaps:
        cv.rectangle(vis, (g.x0, g.y0), (g.x1, g.y1), (0, 0, 255), 3)
        label = f"{g.kind} {g.severity:.2f}"
        cv.putText(vis, label, (g.x0, max(20, g.y0 - 8)), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2)
    out = os.path.join(outdir, f"{name}_annotated.png")
    cv.imwrite(out, vis)
    print(f"[{name}]  {w}x{h}  fg={cv.countNonZero(fg)}  缺口={len(gaps)}")
    for g in gaps:
        print(f"      {g.kind:5s} bbox=({g.x0},{g.y0},{g.x1},{g.y1}) sev={g.severity:.2f}")
    return out, len(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imgs", nargs="*")
    ap.add_argument("-o", "--outdir", default=os.path.join(ROOT, "research", "out"))
    ap.add_argument("--name", nargs="*")
    ap.add_argument("--rtsp", action="store_true", help="抓一帧实时流并加入实验")
    args = ap.parse_args()

    imgs, names = list(args.imgs), list(args.name or [])
    if args.imgs and not args.name:
        names = [os.path.splitext(os.path.basename(p))[0] for p in imgs]

    os.makedirs(args.outdir, exist_ok=True)
    if args.rtsp:
        livef = os.path.join(args.outdir, "_live_frame.png")
        if fetch_live_frame(livef):
            imgs.append(livef)
            names.append("live")

    if not imgs:
        imgs = [os.path.join(DATA, "input", "1749_1333.png"),
                os.path.join(DATA, "input", "1749_0944.png")]
        names = ["day_1333", "day_0944"]

    for p, n in zip(imgs, names):
        if not os.path.exists(p):
            print("skip", p)
            continue
        annotate(cv.imread(p), n, args.outdir)


if __name__ == "__main__":
    main()
