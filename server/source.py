"""RTSP 拉流驱动：后台线程连续取流只留最新帧，处理慢不积压；断流自动重连。

raw I/O 细节（后端选择、FFMPEG 参数、重连节奏）全部封在本层，
上层只见 start()/read()->BGR 帧/stop()。

后端：ffmpeg rawvideo 管道。原因：本机 OpenCV 自带 FFMPEG 解此路 HEVC
持续输出灰帧（Could not find ref with POC，100+帧不恢复），而系统
ffmpeg 解码正常。无 ffmpeg 二进制时回退 cv2.VideoCapture（老路）。
"""
import os
import shutil
import subprocess
import threading
import time

import cv2 as cv
import numpy as np
from loguru import logger

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

RETRY_WAIT_S = 3.0
TIMEOUT_US = 10000000  # socket 停滞即 abort，由本层重连（新版 ffmpeg 叫 -timeout，老 -stimeout 已废弃）


class Reader:
    def __init__(self, url: str):
        self._url = url
        self._frame = None
        self._seq = 0  # 帧序号 (burst_next 去重用, read() 语义不变)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._proc = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kill_proc()

    def read(self):
        """最新帧，无帧返回 None。调用方持副本，线程安全。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def burst_next(self, n: int, timeout_s: float = 30.0):
        """收集 n 帧新帧 (按 seq 去重)；超时抛 TimeoutError (night_lamp burst 用)。"""
        out, t0 = [], time.time()
        with self._lock:
            last = self._seq
        while len(out) < n:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"burst_next超时 {len(out)}/{n}")
            with self._lock:
                seq = self._seq
                fr = None if self._frame is None else self._frame.copy()
            if fr is not None and seq != last:
                last = seq
                out.append(fr)
            else:
                time.sleep(0.02)
        return out

    def _loop(self):
        if shutil.which("ffmpeg"):
            self._loop_pipe()
        else:
            logger.warning("无 ffmpeg，走 cv2 回退（HEVC 可能灰帧）")
            self._loop_cv()

    def _geometry(self):
        """ffprobe 取宽高；失败用 cv2 读一帧看形状（灰帧形状也对）尽量返回；都失败返回 None。"""
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                 "-select_streams", "v:0", "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", self._url], timeout=20).decode().strip().split(",")
            w, h = int(out[0]), int(out[1])
            if w > 0 and h > 0:
                return w, h
        except Exception as e:
            logger.debug("ffprobe 失败: {}", e)
        try:
            cap = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
            ok, frame = cap.read()
            cap.release()
            if ok:
                h, w = frame.shape[:2]
                return w, h
        except Exception as e:
            logger.debug("cv2 形状探测失败: {}", e)
        return None

    def _spawn(self):
        cmd = ["ffmpeg", "-hide_banner", "-v", "error",
               "-rtsp_transport", "tcp", "-timeout", str(TIMEOUT_US),
               "-fflags", "nobuffer", "-i", self._url,
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read_exact(self, n: int):
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _kill_proc(self):
        proc, self._proc = self._proc, None
        if proc:
            try:
                proc.kill()
            except Exception:
                pass

    def _loop_pipe(self):
        geo = None
        while not self._stop.is_set():
            if geo is None:
                geo = self._geometry()
                if geo is None:
                    logger.warning("rtsp 形状探测失败，{}s 后重试: {}", RETRY_WAIT_S, self._url)
                    time.sleep(RETRY_WAIT_S)
                    continue
            w, h = geo
            try:
                self._proc = self._spawn()
            except Exception as e:
                logger.warning("ffmpeg 启动失败({})，{}s 后重试", e, RETRY_WAIT_S)
                time.sleep(RETRY_WAIT_S)
                continue
            logger.info("rtsp 已连接(ffmpeg 管道 {}x{}): {}", w, h, self._url)
            size = w * h * 3
            while not self._stop.is_set():
                raw = self._read_exact(size)
                if raw is None:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            rc = self._proc.poll() if self._proc else None
            self._kill_proc()
            if self._stop.is_set():
                break
            logger.warning("rtsp 断流(rc={})，{}s 后重连", rc, RETRY_WAIT_S)
            time.sleep(RETRY_WAIT_S)

    def _open(self):
        cap = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def _loop_cv(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                logger.warning("rtsp 连接失败，{}s 后重试: {}", RETRY_WAIT_S, self._url)
                time.sleep(RETRY_WAIT_S)
                continue
            logger.info("rtsp 已连接(cv2): {}", self._url)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    logger.warning("rtsp 断流，重连")
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
            cap.release()
            time.sleep(RETRY_WAIT_S)
