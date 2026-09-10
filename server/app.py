"""HTTP 服务：/ 实时页（含缺口状态框），/stream MJPEG，/snapshot.jpg 单帧，/mask.png 裸二值图。

算法线程持续拉流 -> 缺口报警(water_barrier.BarrierAlarm) -> 时序确认(track.py) ->
叠加；HTTP 端点只读最新结果，互不阻塞。由 tmp/server/app.py 重构（slot -> 自主检测）。
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2 as cv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from loguru import logger

from . import render
from .config import load
from .engine import Monitor
from .source import Reader

_params = load()
_p = _params["server"]
_LOG_EVERY = 100  # 处理帧数每达到该值打印一次 fps
_EVIDENCE_DIR = Path("evidence")  # 报警帧落盘目录（文件名带时间+状态）
_EVIDENCE_RESAVE_S = 300.0  # 同一次持续报警最多每 5 分钟补存一张，防刷盘


reader = Reader(_params["rtsp_url"])
algo = _params["water_gap"]

_lock = threading.Lock()
_vis_jpeg = None
_mask = None
_n = 0


def _save_evidence(vis, views):
    """存报警叠加帧：文件名 = 相机_本地时间_状态。失败只记日志，不影响主循环。"""
    try:
        _EVIDENCE_DIR.mkdir(exist_ok=True)
        states = "+".join(sorted({v.state.name for v in views}))
        ts = time.strftime("%Y%m%d-%H%M%S")
        fp = _EVIDENCE_DIR / f"1749_{ts}_{states}.jpg"
        cv.imwrite(str(fp), vis)
        logger.info("报警帧已存: {}", fp)
    except Exception as e:
        logger.warning("报警帧保存失败: {}", e)


def _worker():
    global _vis_jpeg, _mask, _n
    mon = None
    t0, n0 = time.time(), 0
    prev_alarming, last_save = False, 0.0
    while True:
        frame = reader.read()
        if frame is None:
            time.sleep(0.05)
            continue
        if mon is None:
            mon = Monitor(frame.shape, algo)
            logger.info("已加载场景先验与算法参数")
        views = mon.step(frame, time.monotonic())
        vis = render.draw_gaps(render.overlay(frame, mon.fg), views)
        alarming = any(v.state.name == "ALARM" for v in views)
        now = time.monotonic()
        if alarming and (not prev_alarming or now - last_save >= _EVIDENCE_RESAVE_S):
            _save_evidence(vis, views)
            last_save = now
        prev_alarming = alarming
        ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, _p["jpeg_quality"]])
        with _lock:
            _vis_jpeg = buf.tobytes() if ok else None
            _mask = mon.fg
            _n += 1
        if _n - n0 >= _LOG_EVERY:
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
            time.sleep(_p["preview_interval_s"])
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


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
