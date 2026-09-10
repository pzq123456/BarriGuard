"""证据落盘: 按相机分目录, 上升沿存 + 同次持续报警节流补存。"""
import time
from pathlib import Path

import cv2 as cv
from loguru import logger


class EvidenceWriter:
    def __init__(self, root="evidence", resave_s=300.0):
        self._root = Path(root)
        self._resave = resave_s
        self._prev = {}
        self._last = {}

    def maybe_save(self, cam_id, vis, alarming, states, now):
        """报警上升沿或节流到期则存一帧, 返回路径或 None。"""
        if not alarming:
            self._prev[cam_id] = False
            return None
        if self._prev.get(cam_id) and now - self._last.get(cam_id, 0.0) < self._resave:
            return None
        try:
            d = self._root / cam_id
            d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{cam_id}_{time.strftime('%Y%m%d-%H%M%S')}_{states}.jpg"
            cv.imwrite(str(fp), vis)
            logger.info("报警帧已存: {}", fp)
            self._prev[cam_id], self._last[cam_id] = True, now
            return str(fp)
        except Exception as e:
            logger.warning("报警帧保存失败: {}", e)
            return None
