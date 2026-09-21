"""拉真实 RTSP 跑水马，报警时把叠加图存到 output/。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
OUT = REPO / "output"
sys.path.insert(0, str(APP))

import cv2 as cv  # noqa: E402

from server import config, registry, render, worker  # noqa: E402
from server.source import Reader  # noqa: E402


def main(seconds=180.0, camera="1749"):
    OUT.mkdir(exist_ok=True)
    cfg = config.load_runtime(TOOLS.parent / "config.yaml")
    cam = next(c for c in cfg.cameras if c.id == camera)
    spec = cam.algorithms["water_gap"].spec
    calib = worker._load_calibration("water_gap", spec)
    print("confidence=%s alarm_hold_s=%s reconfirm_s=%s rer_threshold=%s"
          % (spec.confidence, spec.alarm_hold_s, spec.reconfirm_s,
             calib["track"]["rer_threshold"]))

    reader = Reader(cam.rtsp_url)
    reader.start()
    frame, t0 = None, time.time()
    while frame is None and time.time() - t0 < 10:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
    if frame is None:
        reader.stop()
        raise SystemExit("no frame from RTSP")
    algo = registry.create("water_gap", frame.shape, calib, cam.id)

    n = saved = 0
    t0 = time.time()
    last_beat = 0.0
    try:
        while time.time() - t0 < seconds:
            f = reader.read()
            if f is None:
                time.sleep(0.02)
                continue
            res = algo.step(f, time.monotonic())
            n += 1
            mask = res.debug.get("roi_mask")
            vis = render.draw_annots(
                render.overlay(f, mask) if mask is not None else f, res.annots)
            vis = render.draw_status(vis, res.debug.get("frame_status", "OK"))
            events = [e for e in res.events if e.kind in ("alarm", "suspected")]
            if events:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for e in events:
                    p = OUT / ("water_%s_%s_%s.jpg"
                               % (stamp, e.kind, e.payload.get("row_id", "")))
                    cv.imwrite(str(p), vis)
                    saved += 1
                    print("saved", p.name, "rer=", e.payload.get("rer"))
            if time.time() - last_beat >= 30:
                cv.imwrite(str(OUT / "water_latest.jpg"), vis)
                last_beat = time.time()
    finally:
        reader.stop()
    print("frames=%d saved_alarms=%d latest=%s" % (n, saved, OUT / "water_latest.jpg"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--camera", default="1749")
    a = ap.parse_args()
    main(a.seconds, a.camera)
