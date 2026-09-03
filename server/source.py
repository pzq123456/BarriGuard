"""RTSP 拉流驱动：后台线程连续取流只留最新帧，处理慢不积压；断流自动重连。

raw I/O 细节（FFMPEG 参数、重连节奏）全部封在本层，上层只见 read() -> BGR 帧。
"""
import os
import threading
import time

import cv2 as cv
from loguru import logger

# 强制 TCP 传输，避免 UDP 丢包花屏；须在 VideoCapture 创建前生效
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RETRY_WAIT_S = 3.0


class Reader:
    def __init__(self, url: str):
        self._url = url
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def read(self):
        """最新帧，无帧返回 None。调用方持副本，线程安全。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _open(self):
        cap = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def _loop(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                logger.warning("rtsp 连接失败，{}s 后重试: {}", RETRY_WAIT_S, self._url)
                time.sleep(RETRY_WAIT_S)
                continue
            logger.info("rtsp 已连接: {}", self._url)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    logger.warning("rtsp 断流，重连")
                    break
                with self._lock:
                    self._frame = frame
            cap.release()
            time.sleep(RETRY_WAIT_S)
