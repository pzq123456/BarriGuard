"""Thin camera wiring -- no abstraction, no factory.

Decision (confirmed): reuse server/source.Reader semantics for RTSP.
Reader lacks what bursts need (frame seq dedup, burst_next(n, timeout),
stop/close), so RTSPSrc below is a verbatim copy of the proven
stream_source.RTSPSrc loop (itself ported from server/source.Reader:
tcp transport, 3s reconnect). When Reader gains seq/burst, delete this copy.

File source lives in tools/replay.FileSrc -- imported here, never duplicated.
"""
import logging
import os
import threading
import time

import cv2 as cv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

LOG = logging.getLogger("night_lamp.camera")
RETRY_WAIT_S = 3.0

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


def open_source(source, video_path=None, rtsp_url=None):
    """source: 'file' -> FileSrc(video_path); 'rtsp' -> started RTSPSrc(rtsp_url)."""
    if source == "file":
        return FileSrc(video_path)
    return RTSPSrc(rtsp_url).start()
