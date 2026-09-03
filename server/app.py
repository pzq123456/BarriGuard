"""HTTP 服务：/ 实时页（含缺口告警框），/stream MJPEG，/snapshot.jpg 单帧，/mask.png 裸二值图。

算法线程持续拉流->分割->slot 缺口检测->叠加，HTTP 端点只读最新结果，互不阻塞。
"""
import threading
import time
from contextlib import asynccontextmanager

import cv2 as cv
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from loguru import logger

from . import config as cfg, render
from .barrier import process
from .gap_detector import (
    RoadColorProfile, SlotTracker, calc_patch_rer_vectorized, evaluate_slot_fast,
)
from .source import Reader

JPEG_QUALITY = 80
STREAM_INTERVAL_S = 0.1
PROCESS_LOG_EVERY = 100

reader = Reader(cfg.CAMERAS[cfg.DEFAULT_CAMERA].rtsp)

_lock = threading.Lock()
_vis_jpeg = None
_mask = None
_n = 0


def _make_confirm(cam, road_rects, slot_mask):
    """RER 确认闭包：中值帧重新分割 + 逐帧重建路色模型，避免陈旧 mask 配对。"""
    def confirm(median_bgr):
        mask = process(median_bgr, cam)
        crops = [median_bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in road_rects]
        min_w = min(c.shape[1] for c in crops)
        profile = RoadColorProfile(np.vstack([c[:, :min_w] for c in crops]))
        return calc_patch_rer_vectorized(median_bgr, slot_mask, mask, profile)
    return confirm


def _init_gap(cam, w, h):
    """首帧初始化：slot 掩膜/状态机/RER 确认闭包，坐标按实流分辨率缩放。"""
    slots = {}
    for name, (rect, base) in cfg.SLOTS_1749.items():
        srect = cfg.scale_rect(rect, w, h)
        sm = np.zeros((h, w), np.uint8)
        x0, y0, x1, y1 = srect
        sm[y0:y1, x0:x1] = 255
        slots[name] = {"mask": sm, "rect": srect, "last_state": None,
                       "tracker": SlotTracker(slot_id=len(slots), base_coverage=base),
                       "confirm": None}
    road_rects = [cfg.scale_rect(r, w, h) for r in cfg.ROAD_ROIS_1749]
    for s in slots.values():
        s["confirm"] = _make_confirm(cam, road_rects, s["mask"])
    return slots


def _update_gap(slots, frame, mask, now):
    """逐 slot 快检 + 状态机推进；返回 {name: (rect, state)} 供渲染。"""
    out = {}
    for name, s in slots.items():
        tr = s["tracker"]
        cond = evaluate_slot_fast(s["mask"], mask, tr.base_coverage)
        st = tr.update(cond, now, s["confirm"])
        tr.push_frame(frame)
        if st is not s["last_state"]:
            logger.info("slot {} : {} -> {}", name, s["last_state"], st.name)
            s["last_state"] = st
        out[name] = (s["rect"], st)
    return out


def _worker():
    global _vis_jpeg, _mask, _n
    cam, slots = None, None
    t0, n0 = time.time(), 0
    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
            continue
        if cam is None:
            h, w = frame.shape[:2]
            cam = cfg.scale_cam(cfg.CAMERAS[cfg.DEFAULT_CAMERA], w, h)
            slots = _init_gap(cam, w, h)
            logger.info("实流 {}x{}，标定坐标已等比缩放", w, h)
        mask = process(frame, cam)
        states = _update_gap(slots, frame, mask, time.monotonic())
        vis = render.draw_slots(render.overlay(frame, mask), states)
        ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with _lock:
            _vis_jpeg = buf.tobytes() if ok else None
            _mask = mask
            _n += 1
        if _n - n0 >= PROCESS_LOG_EVERY:
            logger.info("处理 {} 帧, {:.1f} fps", _n, (_n - n0) / (time.time() - t0))
            t0, n0 = time.time(), _n


def _latest():
    with _lock:
        return _vis_jpeg, _mask


@asynccontextmanager
async def lifespan(_):
    reader.start()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="BarriGuard", lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(
        "<body style='margin:0;background:#111;text-align:center'>"
        "<h3 style='color:#eee;margin:8px'>BarriGuard 1749 水马叠加 + 缺口告警</h3>"
        "<img src='/stream' style='max-width:100%'>"
        "<h3 style='color:#eee;margin:8px'>mask_barrier</h3>"
        "<img src='/mask' style='max-width:100%'></body>")


@app.get("/stream")
async def stream():
    def frames():
        while True:
            jpeg, _ = _latest()
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
            time.sleep(STREAM_INTERVAL_S)
    return StreamingResponse(frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot.jpg")
async def snapshot():
    jpeg, _ = _latest()
    if not jpeg:
        return Response(status_code=503)
    return Response(jpeg, media_type="image/jpeg")


@app.get("/mask.png")
async def mask():
    _, m = _latest()
    if m is None:
        return Response(status_code=503)
    ok, buf = cv.imencode(".png", m)
    return Response(buf.tobytes(), media_type="image/png")
