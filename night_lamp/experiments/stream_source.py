"""Burst取帧源（研究侧独立模块，不依赖 server/）。

FileSrc: 本地视频确定性回放（帧精确 seek），用于可重放实验。
RTSPSrc: 直播流后台线程只留最新帧（处理慢不积压），断流自动重连。
  语义移植自 server/source.py::Reader（tcp 传输、3s 重连节奏保持一致），
  日志改用标准库（研究脚本零新增依赖：仅 cv2/numpy）。
"""
import logging
import os
import threading
import time

import cv2 as cv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

LOG = logging.getLogger("night_lamp.source")
RETRY_WAIT_S = 3.0


class FileSrc:
    """确定性文件源：burst_at(f0, n) -> n 帧 BGR（不足 n 帧抛错）。"""

    def __init__(self, path):
        self.path = path
        self._cap = cv.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"打不开视频: {path}")
        self._fps = float(self._cap.get(cv.CAP_PROP_FPS))
        self._total = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))

    def fps(self):
        return self._fps

    def total(self):
        return self._total

    def burst_at(self, f0, n):
        self._cap.set(cv.CAP_PROP_POS_FRAMES, f0)
        out = []
        for _ in range(n):
            ok, fr = self._cap.read()
            if not ok:
                break
            out.append(fr)
        if len(out) != n:
            raise RuntimeError(f"short_read f0={f0} got={len(out)}/{n}")
        return out

    def frame_at(self, f0):
        """单帧（mid-interval snapshot 用，不影响 burst 语义）。"""
        self._cap.set(cv.CAP_PROP_POS_FRAMES, f0)
        ok, fr = self._cap.read()
        if not ok:
            raise RuntimeError(f"seek_read_fail f0={f0}")
        return fr

    def close(self):
        self._cap.release()


class RTSPSrc:
    """直播源：后台线程连续取流，只留最新帧；burst_next(n) 收 n 个新帧。"""

    def __init__(self, url, retry_wait_s=RETRY_WAIT_S):
        self._url = url
        self._wait = retry_wait_s
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def read(self):
        """最新帧副本，无帧返回 None（同 server/source.Reader 语义）。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def burst_next(self, n, timeout_s=30.0):
        """收 n 个连续新帧（按 seq 去重）；超时抛 TimeoutError（调用方记 failed）。"""
        out, last, t0 = [], -1, time.time()
        with self._lock:
            last = self._seq
        while len(out) < n:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"burst_next timeout got={len(out)}/{n}")
            with self._lock:
                seq, fr = self._seq, (None if self._frame is None else self._frame.copy())
            if fr is not None and seq != last:
                last = seq
                out.append(fr)
            else:
                time.sleep(0.02)
        return out

    def stop(self):
        self._stop.set()

    def close(self):
        self.stop()

    def _open(self):
        cap = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def _loop(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                LOG.warning("rtsp 连接失败，%ss 后重试", self._wait)
                time.sleep(self._wait)
                continue
            LOG.info("rtsp 已连接")
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    LOG.warning("rtsp 断流，重连")
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            cap.release()
            time.sleep(self._wait)
