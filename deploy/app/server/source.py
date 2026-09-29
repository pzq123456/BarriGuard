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

# 灰帧自愈：cv2 / Debian 7.1 / essentials 构建解 1749 的 HEVC 会持续输出灰帧
# （std≈2~7，正常画面 40~70）。持续 GREY_RUN_LIMIT 帧近灰就判解码失败并重连，
# 不再静默失明。
FFMPEG_ENV = "BARRIGUARD_FFMPEG"  # 显式指定 ffmpeg 路径，优先于 PATH
GREY_STD_TH = 12.0
GREY_RUN_LIMIT = 125
GREY_SUBSAMPLE = 8


def _flag_supported(path: str, flags: list) -> bool:
    """跑一个 1 帧的 lavfi 空转，确认该 ffmpeg 认这组输出选项。"""
    cmd = [path, "-hide_banner", "-v", "error", "-f", "lavfi",
           "-i", "testsrc=d=0.04", "-frames:v", "1"] + list(flags) + [
        "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def _passthrough_args(path: str) -> list:
    """帧率直通参数：新版只有 -fps_mode（-vsync 已被移除，传了直接 rc=8），
    老版用 -vsync；都探测不到则不传。"""
    for flags in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        if _flag_supported(path, flags):
            return flags
    return []


def _version_of(path: str) -> str:
    """ffmpeg 版本首行，仅用于把解码后端写进日志。"""
    try:
        out = subprocess.check_output([path, "-version"],
                                      stderr=subprocess.DEVNULL, timeout=10)
        return out.decode("utf-8", "replace").splitlines()[0].strip()
    except Exception:
        return "unknown"


class Reader:
    def __init__(self, url: str):
        self._url = url
        self._frame = None
        self._seq = 0  # 帧序号 (burst_next 去重用, read() 语义不变)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._proc = None
        self._connected = False
        self._started_mono = None
        self._last_frame_mono = None
        self._reconnects = 0
        self._grey_run = 0
        self._backend = None
        self._ffmpeg = None
        self._passthrough = []

    def start(self):
        self._started_mono = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._kill_proc()
        self.join()

    def join(self, timeout: float = 5.0):
        """等采集线程收敛；用于优雅停机，不阻塞在已死线程上。"""
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def health(self) -> dict:
        """拉流健康快照；Worker 据此判定断流/恢复。"""
        with self._lock:
            last = self._last_frame_mono
            return {
                "connected": self._connected,
                "started_mono": self._started_mono,
                "last_frame_mono": last,
                "seconds_since_frame": (None if last is None
                                        else time.monotonic() - last),
                "reconnects": self._reconnects,
                "backend": self._backend,
                "grey_run": self._grey_run,
            }

    def _mark_frame(self):
        with self._lock:
            self._last_frame_mono = time.monotonic()
            self._connected = True

    def _mark_connected(self, value: bool):
        with self._lock:
            self._connected = value
            if not value:
                self._reconnects += 1

    def _is_grey(self, frame) -> bool:
        """抽样估灰度 std；近常量即判灰帧（正常画面 40~70，灰帧 <10）。"""
        small = frame[::GREY_SUBSAMPLE, ::GREY_SUBSAMPLE]
        gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY) if small.ndim == 3 else small
        return float(gray.std()) < GREY_STD_TH

    def read(self):
        """最新帧，无帧返回 None。调用方持副本，线程安全。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def seq(self) -> int:
        """当前帧序号；无新帧时 read() 会重复返回同一帧，据此去重。"""
        with self._lock:
            return self._seq

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
        if not self._is_stream():
            self._backend = "cv2-local"
            logger.info("本地输入(cv2): {}", self._url)
            self._loop_cv()
            return
        self._ffmpeg = self._resolve_ffmpeg()
        if self._ffmpeg:
            self._backend = "ffmpeg"
            self._passthrough = _passthrough_args(self._ffmpeg)
            logger.info("解码后端 ffmpeg: {} | {} | passthrough={}",
                        self._ffmpeg, _version_of(self._ffmpeg),
                        self._passthrough or "-")
            self._loop_pipe()
        else:
            self._backend = "cv2"
            logger.warning("无 ffmpeg，走 cv2 回退：1749 的 HEVC 会持续灰帧，"
                           "必须给镜像装 ffmpeg（见 deploy/Dockerfile）")
            self._loop_cv()

    def _resolve_ffmpeg(self):
        """显式 env > PATH；都找不到返回 None（回退 cv2）。"""
        override = os.environ.get(FFMPEG_ENV, "").strip()
        if override:
            if os.path.isfile(override):
                return override
            logger.warning("{}={} 不存在，回退 PATH 查找", FFMPEG_ENV, override)
        found = shutil.which("ffmpeg")
        return found or None

    def _is_stream(self) -> bool:
        return self._url.lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://", "rtmp://"))

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
        cmd = [self._ffmpeg or "ffmpeg", "-hide_banner", "-v", "error",
               "-rtsp_transport", "tcp", "-timeout", str(TIMEOUT_US),
               "-fflags", "nobuffer", "-i", self._url,
               "-f", "rawvideo", "-pix_fmt", "bgr24", *self._passthrough, "-"]
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
            self._mark_connected(True)
            size = w * h * 3
            while not self._stop.is_set():
                raw = self._read_exact(size)
                if raw is None:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
                if self._is_grey(frame):
                    self._grey_run += 1
                    if self._grey_run >= GREY_RUN_LIMIT:
                        logger.error("画面持续近灰(run={})，判解码失败并重连: {}",
                                     self._grey_run, self._url)
                        self._grey_run = 0
                        break
                else:
                    self._grey_run = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._last_frame_mono = time.monotonic()
                    self._connected = True
            rc = self._proc.poll() if self._proc else None
            self._kill_proc()
            if self._stop.is_set():
                break
            self._mark_connected(False)
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
            self._mark_connected(True)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    logger.warning("rtsp 断流，重连")
                    break
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._last_frame_mono = time.monotonic()
                    self._connected = True
            cap.release()
            self._mark_connected(False)
            time.sleep(RETRY_WAIT_S)
