"""HTTP 服务：/ 相机列表，/stream/{id} MJPEG，/snapshot/{id}.jpg 单帧。

算法调试图层：/debug/{id}/{layer}.png（层名由算法定，如 roi_mask，按需加）。
报警事件流：/events/{id}（统一 Event schema，转发模块以后直接消费它）。
不带 id 的旧路由指向首个相机（兼容）。
"""
import time
from contextlib import asynccontextmanager

import cv2 as cv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .config import load
from .evidence import EvidenceWriter
from .worker import CameraWorker

_params = load()
_p = _params["server"]
_evidence = EvidenceWriter(resave_s=_p.get("evidence_resave_s", 300.0))
_workers = {c["id"]: CameraWorker(c, _evidence, _p["jpeg_quality"])
            for c in _params["cameras"]}
_first = next(iter(_workers.values()))


@asynccontextmanager
async def lifespan(_):
    for w in _workers.values():
        w.start()
    yield


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
        time.sleep(_p["preview_interval_s"])


@app.get("/stream/{cam_id}")
async def stream(cam_id: str):
    w = _workers.get(cam_id)
    if w is None:
        return Response(status_code=404)
    return StreamingResponse(_frames(w),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream")
async def stream_legacy():
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
    jpeg, _ = _first.latest()
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
    return _png(_first.debug_layer("roi_mask"))


@app.get("/events/{cam_id}")
async def events(cam_id: str):
    """近期报警事件（统一 schema，转发模块的消费口）。status 非 OK 表示盲区。"""
    w = _workers.get(cam_id)
    if w is None:
        return Response(status_code=404)
    return {"camera_id": cam_id, "status": w.status(), "events": w.events()}
