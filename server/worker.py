"""单相机工作循环: Reader 拉流 -> 各算法 -> 通用渲染/证据/事件。

HTTP 端点只读最新结果, 互不阻塞。无启用算法时原帧直通 (纯视频流)。
本模块不认识任何具体算法：构造经 registry，渲染只认 Annotation，
证据/事件只认 Event 通用字段 (转发模块以后直接消费 events)。
"""
import threading
import time
from collections import deque

import cv2 as cv
from loguru import logger

from . import registry, render
from .evidence import EvidenceWriter
from .source import Reader

LOG_EVERY = 100
EVENTS_KEPT = 200


class CameraWorker:
    def __init__(self, cam: dict, evidence: EvidenceWriter, jpeg_quality: int):
        self.id = cam["id"]
        self.name = cam.get("name") or cam["id"]
        self._specs = cam["algos"]
        self._reader = Reader(cam["rtsp_url"])
        self._evidence = evidence
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._vis = None
        self._debug = {}
        self._status = "OK"
        self._events = deque(maxlen=EVENTS_KEPT)
        self._n = 0
        self._algos = {}
        self._warned_periodic = False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cam-{self.id}")

    @property
    def algo_names(self):
        return sorted(self._specs)

    def start(self):
        self._reader.start()
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._vis, dict(self._debug)

    def status(self):
        with self._lock:
            return self._status

    def debug_layer(self, layer):
        with self._lock:
            m = self._debug.get(layer)
            return None if m is None else m.copy()

    def events(self):
        with self._lock:
            return list(self._events)

    def _loop(self):
        t0, n0 = time.time(), 0
        while True:
            frame = self._reader.read()
            if frame is None:
                time.sleep(0.05)
                continue
            if not self._algos and self._specs:
                for name, calib in self._specs.items():
                    self._algos[name] = registry.create(name, frame.shape,
                                                        calib, self.id)
                logger.info("[{}] 已加载算法: {}", self.id, sorted(self._algos))
            annots, debug, events = [], {}, []
            status = "OK"
            for name, algo in self._algos.items():
                if algo.cadence != "per_frame":
                    if not self._warned_periodic:
                        logger.warning("[{}] {} 为 periodic，工作循环暂跳过", self.id, name)
                        self._warned_periodic = True
                    continue
                res = algo.step(frame, time.monotonic())
                annots += res.annots
                debug.update(res.debug)
                events += res.events
                if isinstance(res.debug.get("frame_status"), str):
                    status = res.debug["frame_status"]
            mask = debug.get("roi_mask")
            vis = render.draw_annots(
                render.overlay(frame, mask) if mask is not None else frame, annots)
            vis = render.draw_status(vis, status)
            alarming = any(e.kind == "alarm" for e in events)
            states = "+".join(sorted({a.level for a in annots}))
            now = time.monotonic()
            path = self._evidence.maybe_save(self.id, vis, alarming, states, now)
            if path:
                for e in events:
                    if e.kind == "alarm" and e.evidence_path is None:
                        e.evidence_path = path
            ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, self._quality])
            with self._lock:
                self._vis = buf.tobytes() if ok else None
                self._debug = debug
                self._status = status
                self._events.extend({"ts": e.ts, "algo": e.algo, "kind": e.kind,
                                     "payload": e.payload,
                                     "evidence_path": e.evidence_path} for e in events)
                self._n += 1
            if self._n - n0 >= LOG_EVERY:
                logger.info("[{}] 处理 {} 帧, {:.1f} fps", self.id, self._n,
                            (self._n - n0) / (time.time() - t0))
                t0, n0 = time.time(), self._n
