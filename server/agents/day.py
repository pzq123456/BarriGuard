"""白昼水马运行时封装（Wave 1 Agent C）。

职责边界（严格）：
  1. 每帧透传给现有 ``WaterGapAlgorithm.step``，算法零改动；
  2. 保留最近一帧与最近一次 ``AlgoResult``，仅供快照出图；
  3. ``snapshot()`` 用 ``cv.imencode`` 在**内存**里产出 JPEG 并包成 ``Report``。

不落盘；告警节流不在此（由 server/reporting.py 的 Reporter 负责）。
"""
import threading
from datetime import datetime

import cv2 as cv

from .. import render
from ..algo import AlgoResult, Report

JPEG_QUALITY = 90


class DayRunner:
    """单相机白昼水马：与 ``CameraSpec`` / ``WaterGapSpec`` / ``AlgoResult`` 契约对齐。"""

    def __init__(self, camera, spec, algo):
        self.camera = camera
        self.spec = spec
        self.algo = algo
        self._lock = threading.Lock()
        self._frame = None
        self._result = None

    def on_frame(self, frame_bgr, ts_mono: float) -> AlgoResult:
        """透传一帧给现有算法，并记住最近帧/结果供 ``snapshot`` 使用。"""
        res = self.algo.step(frame_bgr, ts_mono)
        with self._lock:
            self._frame = frame_bgr
            self._result = res
        return res

    def snapshot(self, ts_wall: datetime) -> Report | None:
        """把最近一帧按当前 annots 渲染成 JPEG，包成 ``gap_overlay`` Report。

        尚无任何帧时返回 None。图片只在内存中编码，不写磁盘。
        """
        with self._lock:
            frame = self._frame
            res = self._result
        if frame is None:
            return None

        annots = res.annots if res is not None else []
        mask = res.debug.get("roi_mask") if res is not None else None
        status = "ok"
        if res is not None:
            fs = res.debug.get("frame_status")
            if isinstance(fs, str):
                status = fs

        vis = render.overlay(frame, mask) if mask is not None else frame
        vis = render.draw_annots(vis, annots)
        ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        return Report(
            camera=self.camera.id,
            algorithm=getattr(self.algo, "name", "water_gap"),
            report_type="gap_overlay",
            created_at=ts_wall.isoformat(),
            image_jpeg=buf.tobytes(),
            metadata={"annots": len(annots), "status": status,
                      "rows": sorted({a.label for a in annots})},
            status="ok",
        )
