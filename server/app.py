"""HTTP 服务：/ 实时页，/stream MJPEG 实时叠加，/snapshot.jpg 单帧，/mask.png 裸二值图。

算法线程持续拉流->分割->叠加，HTTP 端点只读最新结果，互不阻塞。
"""
import threading
import time
from contextlib import asynccontextmanager

import cv2 as cv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from loguru import logger

from . import config as cfg, render
from .barrier import process
from .source import Reader

JPEG_QUALITY = 80
STREAM_INTERVAL_S = 0.1
PROCESS_LOG_EVERY = 100

reader = Reader(cfg.CAMERAS[cfg.DEFAULT_CAMERA].rtsp)

_lock = threading.Lock()
_vis_jpeg = None
_mask = None
_n = 0


def _worker():
    global _vis_jpeg, _mask, _n
    cam, t0, n0 = None, time.time(), 0
    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
            continue
        if cam is None:
            h, w = frame.shape[:2]
            cam = cfg.scale_cam(cfg.CAMERAS[cfg.DEFAULT_CAMERA], w, h)
            logger.info("实流 {}x{}，标定坐标已等比缩放", w, h)
        mask = process(frame, cam)
        vis = render.overlay(frame, mask)
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
        "<h3 style='color:#eee;margin:8px'>BarriGuard 1749 水马二值叠加</h3>"
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
