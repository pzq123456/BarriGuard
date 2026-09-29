"""HTTP 服务：/ 相机列表，/stream/{id} MJPEG，/snapshot/{id}.jpg 单帧。

算法调试图层：/debug/{id}/{layer}.png（层名由算法定，如 roi_mask，按需加）。
报警事件流：/events/{id}（统一 Event schema，转发模块以后直接消费它）。
不带 id 的旧路由指向首个相机（兼容）。

Wave 1：配置走 `server.config.load_runtime`，编排走 `server.worker.Runtime`
（Scheduler 门控 day/night 算法与夜灯 finalize 上报）。
"""
import os
import time
from contextlib import asynccontextmanager

import cv2 as cv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .config import load_runtime
from .logging_setup import configure as _configure_logging, startup_banner
from .worker import Runtime

_cfg = load_runtime(os.environ.get("BARRIGUARD_CONFIG") or None)
_p = _cfg.server if isinstance(_cfg.server, dict) else {}
# 直接 uvicorn server.app:app 启动时也要落盘日志；与 __main__ 重复调用为 no-op。
_configure_logging(_p.get("log_level", "warning"), _p.get("output_dir"),
                   _p.get("log_dir"), banner=startup_banner(_cfg))
_runtime = Runtime(_cfg)
_workers = _runtime.workers
_first = next(iter(_workers.values()), None)


@asynccontextmanager
async def lifespan(_):
    _runtime.start()
    try:
        yield
    finally:
        _runtime.stop()


app = FastAPI(title="BarriGuard", lifespan=lifespan)


@app.get("/")
async def index():
    items = "".join(
        f"<div style='margin:12px'><h3 style='color:#eee'>{w.name} ({w.id})</h3>"
        f"<img src='/stream/{w.id}' style='max-width:100%'><br>"
        f"<a style='color:#888' href='/events/{w.id}'>events</a></div>"
        for w in _workers.values())
    return HTMLResponse(
        "<body style='margin:0;background:#111;text-align:center'>" + items + "</body>")


@app.get("/cameras")
async def cameras():
    return [{"id": w.id, "name": w.name, "algos": w.algo_names}
            for w in _workers.values()]


def _frames(w):
    while True:
        jpeg, _ = w.latest()
        if jpeg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
        time.sleep(_p.get("preview_interval_s", 0.1))


@app.get("/stream/{cam_id}")
async def stream(cam_id: str):
    w = _workers.get(cam_id)
    if w is None:
        return Response(status_code=404)
    return StreamingResponse(_frames(w),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream")
async def stream_legacy():
    if _first is None:
        return Response(status_code=503)
    return StreamingResponse(_frames(_first),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot/{cam_id}.jpg")
async def snapshot(cam_id: str):
    w = _workers.get(cam_id)
    jpeg, _ = w.latest() if w else (None, None)
    if not jpeg:
        return Response(status_code=503)
    return Response(jpeg, media_type="image/jpeg")


@app.get("/snapshot.jpg")
async def snapshot_legacy():
    jpeg, _ = _first.latest() if _first else (None, None)
    if not jpeg:
        return Response(status_code=503)
    return Response(jpeg, media_type="image/jpeg")


def _png(img):
    if img is None:
        return Response(status_code=503)
    ok, buf = cv.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png")


@app.get("/debug/{cam_id}/{layer}.png")
async def debug(cam_id: str, layer: str):
    """算法调试图层（如 roi_mask）。层名由算法定，server 原样 serving。"""
    w = _workers.get(cam_id)
    if w is None:
        return Response(status_code=404)
    return _png(w.debug_layer(layer))


@app.get("/mask/{cam_id}.png")
async def mask(cam_id: str):
    w = _workers.get(cam_id)
    return _png(w.debug_layer("roi_mask") if w else None)


@app.get("/mask.png")
async def mask_legacy():
    return _png(_first.debug_layer("roi_mask") if _first else None)


@app.get("/events/{cam_id}")
async def events(cam_id: str):
    """近期报警事件（统一 schema，转发模块的消费口）。status 非 OK 表示盲区。"""
    w = _workers.get(cam_id)
    if w is None:
        return Response(status_code=404)
    return {"camera_id": cam_id, "status": w.status(), "events": w.events()}
