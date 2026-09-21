"""告警帧服务端落盘：callback 之外的持久副本。

Reporter 的 callback 是 best-effort（队列满/消费端挂即丢），因此告警帧必须
先在本机留下 jpg + json，消费端不可达时也能事后追溯。root 为空则 no-op。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")
_ALARM_DIR = "alerts"


def _safe(text, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", str(text or "")).strip("._-")
    return cleaned or fallback


class AlarmStore:
    """把每条告警写成 ``<root>/alerts/<camera>/<ts>_<algo>_<target>.json/.jpg``。"""

    def __init__(self, root):
        root = str(root or "").strip()
        self._root = (Path(root) / _ALARM_DIR) if root else None

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def save(self, camera_id, algo, target, payload, image_jpeg,
             received_at: datetime | None = None):
        if not self.enabled:
            return None
        ts = received_at or datetime.now(timezone.utc)
        cam = _safe(camera_id, "unknown")
        outdir = self._root / cam
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"{ts.strftime('%Y%m%d_%H%M%S_%f')}_{_safe(algo, 'alarm')}_{_safe(target, 'na')}"
        jpg = outdir / f"{stem}.jpg"
        meta = outdir / f"{stem}.json"
        try:
            if image_jpeg:
                jpg.write_bytes(image_jpeg)
            record = {
                "camera_id": camera_id,
                "algo": algo,
                "target": target,
                "received_at": ts.isoformat(),
                "frame": jpg.name if image_jpeg else None,
                **(payload or {}),
            }
            meta.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
        except Exception:
            logger.exception("[alarm_store] 落盘失败 {}", stem)
            return None
        return str(meta)
