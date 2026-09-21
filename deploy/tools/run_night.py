"""用昨晚的录像离线跑夜灯，按配置出热力图到 output/。"""
from __future__ import annotations

import json
import sys
import time
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


def _auto_day(camera):
    d = REPO / "data" / camera
    for p in sorted(d.glob("*day.jpg")):
        return p
    return None


def main(frames=3000, camera="1749", video=None, day=None):
    OUT.mkdir(exist_ok=True)
    cfg = config.load_runtime(TOOLS.parent / "config.yaml")
    cam = next(c for c in cfg.cameras if c.id == camera)
    spec = cam.algorithms["night_lamp"].spec
    print("camera=%s night_gate=%s overnight=%s"
          % (camera, spec.night_gate, spec.overnight))

    video = Path(video) if video else REPO / "tmp" / "1002490_20260918_1900_night.mp4"
    day = Path(day) if day else _auto_day(camera)
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)

    session = NightSession(cam.id, spec)
    step = max(int(spec.sampling.interval_ms), 1) / 1000.0
    ts = 0.0
    n = 0
    t0 = time.time()
    while n < frames:
        ok, f = cap.read()
        if not ok:
            break
        session.accumulate(f, ts)
        ts += step
        n += 1
    cap.release()

    base = cv.imread(str(day)) if day else None
    rep = session.finalize(base, datetime.now(timezone.utc))
    p = OUT / ("night_heatmap_%s.jpg" % camera)
    if rep.image_jpeg:
        p.write_bytes(rep.image_jpeg)
    md = rep.metadata
    keys = ("alignment_status", "day_alignment", "image_mode", "n_cand",
            "n_cand_total", "n_flash", "n_reflect", "night_state",
            "night_qualified", "candidate_overflow", "memory_mb", "warmup")
    meta_p = OUT / ("night_heatmap_%s.json" % camera)
    meta_p.write_text(json.dumps(
        {"camera": camera, "video": video.name, "frames": n,
         "status": rep.status, "report_type": rep.report_type,
         "image": p.name, "metadata": {k: md.get(k) for k in keys}},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("video=%s day=%s frames=%d elapsed=%.0fs status=%s type=%s image=%s"
          % (video.name, day.name if day else None, n, time.time() - t0,
             rep.status, rep.report_type, p.name))
    print("alignment_status=%s day_alignment=%s image_mode=%s n_cand=%s "
          "n_flash=%s night_qualified=%s"
          % (md.get("alignment_status"), md.get("day_alignment"),
             md.get("image_mode"), md.get("n_cand"), md.get("n_flash"),
             md.get("night_qualified")))
    print("meta ->", meta_p.name)
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
