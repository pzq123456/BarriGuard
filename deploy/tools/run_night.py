"""用昨晚的录像离线跑夜灯，按配置出热力图到 output/。"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
OUT = REPO / "output"
sys.path.insert(0, str(APP))

import cv2 as cv  # noqa: E402

from night_lamp.session import NightSession  # noqa: E402
from server import config  # noqa: E402


def main(frames=2000, camera="1749", video=None, day=None):
    OUT.mkdir(exist_ok=True)
    cfg = config.load_runtime(TOOLS.parent / "config.yaml")
    cam = next(c for c in cfg.cameras if c.id == camera)
    spec = cam.algorithms["night_lamp"].spec
    print("night_gate=%s overnight=%s" % (spec.night_gate, spec.overnight))

    video = Path(video) if video else REPO / "tmp" / "1002490_20260918_1900_night.mp4"
    day = Path(day) if day else REPO / "data" / "1749" / "1749_20260917_155840_day.jpg"
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)

    session = NightSession(cam.id, spec)
    step = max(int(spec.sampling.interval_ms), 1) / 1000.0
    ts = n = 0.0
    while int(n) < frames:
        ok, f = cap.read()
        if not ok:
            break
        session.accumulate(f, ts)
        ts += step
        n += 1
    cap.release()

    base = cv.imread(str(day))
    rep = session.finalize(base, datetime.now(timezone.utc))
    p = OUT / "night_heatmap.jpg"
    if rep.image_jpeg:
        p.write_bytes(rep.image_jpeg)
    md = rep.metadata
    print("frames=%d status=%s type=%s image=%s" % (int(n), rep.status,
                                                   rep.report_type, p.name))
    print("n_cand=%s n_flash=%s alignment=%s night_qualified=%s"
          % (md.get("n_cand"), md.get("n_flash"), md.get("alignment"),
             md.get("night_qualified")))
    session.release()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--video", default=None)
    ap.add_argument("--day", default=None)
    a = ap.parse_args()
    main(a.frames, a.camera, a.video, a.day)
