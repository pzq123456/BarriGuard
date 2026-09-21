"""拉真实 RTSP 跑水马：定时存运行过程图到 output/water_run/，报警时另存告警帧。"""
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


def main(seconds=300.0, camera="1749", snap_every=45.0):
    run_dir = OUT / "water_run"
    run_dir.mkdir(parents=True, exist_ok=True)
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
    while frame is None and time.time() - t0 < 15:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
    if frame is None:
        reader.stop()
        raise SystemExit("no frame from RTSP")
    algo = registry.create("water_gap", frame.shape, calib, cam.id)
    print("frame shape", frame.shape)

    n = saved = 0
    t0 = time.time()
    last_snap = 0.0
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
            elapsed = time.time() - t0
            if elapsed - last_snap >= snap_every or last_snap == 0.0:
                p = run_dir / ("%s_%s.jpg" % (camera,
                                              time.strftime("%H%M%S")))
                cv.imwrite(str(p), vis)
                last_snap = elapsed
                print("snap t=%.0fs frames=%d fps=%.1f -> %s"
                      % (elapsed, n, n / max(elapsed, 0.1), p.name))
            for e in res.events:
                if e.kind in ("alarm", "suspected"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    q = OUT / ("water_%s_%s_%s.jpg"
                               % (stamp, e.kind, e.payload.get("row_id", "")))
                    cv.imwrite(str(q), vis)
                    saved += 1
                    print("ALARM", e.kind, "row=", e.payload.get("row_id"),
                          "rer=", e.payload.get("rer"), "->", q.name)
    finally:
        reader.stop()
    print("done frames=%d fps=%.1f alarm_images=%d snapshots=%s"
          % (n, n / max(time.time() - t0, 0.1), saved, run_dir))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--snap-every", type=float, default=45.0)
    a = ap.parse_args()
    main(a.seconds, a.camera, a.snap_every)
