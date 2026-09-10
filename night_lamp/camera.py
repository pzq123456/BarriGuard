"""Thin camera wiring -- no I/O logic lives here.

RTSP backend: server/source.Reader (single owner of the ffmpeg rawvideo
pipe, reconnect, seq/burst machinery). This module only adapts its API
(start() must return self, main.py calls close()) and re-exports FileSrc.

File source lives in tools/replay.FileSrc -- imported here, never duplicated.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.source import Reader  # noqa: E402  (needs _ROOT above; loguru is a project dep)

try:
    from tools.replay import FileSrc
except ImportError:  # direct `python camera.py` from night_lamp/
    from replay import FileSrc  # noqa: F401  (tools/replay.py on sys.path)


class RTSPSource:
    """Adapter over server Reader: identical behavior, main.py-shaped API."""

    def __init__(self, url):
        self._r = Reader(url)

    def start(self):
        self._r.start()
        return self

    def read(self):
        return self._r.read()

    def burst_next(self, n, timeout_s=30.0):
        return self._r.burst_next(n, timeout_s=timeout_s)

    def stop(self):
        self._r.stop()

    def close(self):
        self._r.stop()


def open_source(source, video_path=None, rtsp_url=None):
    """source: 'file' -> FileSrc(video_path); 'rtsp' -> started RTSPSource(rtsp_url)."""
    if source == "file":
        return FileSrc(video_path)
    return RTSPSource(rtsp_url).start()
