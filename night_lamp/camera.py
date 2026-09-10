"""Thin camera wiring -- no abstraction, no factory.

RTSP backend: ffmpeg rawvideo pipe, ported from server/source.Reader.
Reason: the bundled OpenCV FFmpeg outputs gray frames for this HEVC
stream ("Could not find ref with POC", tens of seconds before the first
keyframe reaches the decoder), while system ffmpeg delivers the first
frame as a real picture. Falls back to cv2.VideoCapture when no ffmpeg
binary exists (gray startup frames possible, logged as warning).

RTSPSrc keeps the burst machinery server Reader lacks (frame seq dedup,
burst_next(n, timeout), stop/close). When Reader gains seq/burst, delete
this copy.

File source lives in tools/replay.FileSrc -- imported here, never duplicated.
"""
import logging
import os
import shutil
import subprocess
import threading
import time

import cv2 as cv
import numpy as np

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

LOG = logging.getLogger("night_lamp.camera")
RETRY_WAIT_S = 3.0
TIMEOUT_US = 10000000  # socket stall aborts, reconnect here (same as server Reader)

try:
    from tools.replay import FileSrc
except ImportError:  # direct `python camera.py` from night_lamp/
    from replay import FileSrc  # noqa: F401  (tools/replay.py on sys.path)


class RTSPSrc:
    """Live source: background thread keeps latest frame; burst_next(n) collects n new frames."""

    def __init__(self, url, retry_wait_s=RETRY_WAIT_S):
        self._url = url
        self._wait = retry_wait_s
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._proc = None
        self._geo = None

    def start(self):
        self._thread.start()
        return self

    def read(self):
        """Latest frame copy, None if none yet (same as server/source.Reader)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def burst_next(self, n, timeout_s=30.0):
        """Collect n consecutive new frames (seq dedup); TimeoutError on timeout."""
        out, last, t0 = [], -1, time.time()
        with self._lock:
            last = self._seq
        while len(out) < n:
            if time.time() - t0 > timeout_s:
                raise TimeoutError("burst_next timeout got=%s/%s" % (len(out), n))
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
        if shutil.which("ffmpeg"):
            self._loop_pipe()
        else:
            LOG.warning("无 ffmpeg，走 cv2 回退（HEVC 可能灰帧）")
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
            LOG.debug("ffprobe 失败: %s", e)
        try:
            cap = cv.VideoCapture(self._url, cv.CAP_FFMPEG)
            ok, frame = cap.read()
            cap.release()
            if ok:
                h, w = frame.shape[:2]
                return w, h
        except Exception as e:
            LOG.debug("cv2 形状探测失败: %s", e)
        return None

    def _spawn(self):
        cmd = ["ffmpeg", "-hide_banner", "-v", "error",
               "-rtsp_transport", "tcp", "-timeout", str(TIMEOUT_US),
               "-fflags", "nobuffer", "-i", self._url,
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read_exact(self, n):
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
        while not self._stop.is_set():
            if self._geo is None:
                self._geo = self._geometry()
                if self._geo is None:
                    LOG.warning("rtsp 形状探测失败，%ss 后重试: %s", self._wait, self._url)
                    time.sleep(self._wait)
                    continue
            w, h = self._geo
            try:
                self._proc = self._spawn()
            except Exception as e:
                LOG.warning("ffmpeg 启动失败(%s)，%ss 后重试", e, self._wait)
                time.sleep(self._wait)
                continue
            LOG.info("rtsp 已连接(ffmpeg 管道 %dx%d)", w, h)
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
            LOG.warning("rtsp 断流(rc=%s)，%ss 后重连", rc, self._wait)
            time.sleep(self._wait)

    def _loop_cv(self):
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                LOG.warning("rtsp 连接失败，%ss 后重试", self._wait)
                time.sleep(self._wait)
                continue
            LOG.info("rtsp 已连接(cv2 回退)")
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


def open_source(source, video_path=None, rtsp_url=None):
    """source: 'file' -> FileSrc(video_path); 'rtsp' -> started RTSPSrc(rtsp_url)."""
    if source == "file":
        return FileSrc(video_path)
    return RTSPSrc(rtsp_url).start()
