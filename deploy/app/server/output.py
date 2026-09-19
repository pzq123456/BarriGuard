"""Report 落盘：把每份 Report 的 JPEG + metadata JSON 写到 host-mounted 目录。

host 侧目录（docker-compose ``./data:/app/data``）因此可直接检查。仅标准库；
写失败只记日志，绝不冒泡到分析循环。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")


def _safe(text, fallback: str) -> str:
    cleaned = _UNSAFE.sub("_", str(text or "")).strip("._-")
    return cleaned or fallback


def _stamp(created_at) -> str:
    """ISO-8601 -> compact ``YYYYMMDDHHMMSS`` for stable filenames."""
    digits = re.sub(r"[^0-9]", "", str(created_at or ""))
    return digits[:14] or "unknown"


class ReportStore:
    """Write reports under ``root/<camera>/<camera>_<ts>_<type>_<status>.*``."""

    def __init__(self, root):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, report) -> str:
        camera = _safe(getattr(report, "camera", ""), "unknown")
        report_type = _safe(getattr(report, "report_type", ""), "report")
        status = _safe(getattr(report, "status", ""), "ok")
        base = f"{camera}_{_stamp(getattr(report, 'created_at', ''))}_" \
               f"{report_type}_{status}"
        outdir = self._root / camera
        outdir.mkdir(parents=True, exist_ok=True)
        jpg = outdir / (base + ".jpg")
        jso = outdir / (base + ".json")

        image = getattr(report, "image_jpeg", None)
        if image:
            jpg.write_bytes(image)
        payload = {
            "camera": getattr(report, "camera", None),
            "algorithm": getattr(report, "algorithm", None),
            "report_type": getattr(report, "report_type", None),
            "created_at": getattr(report, "created_at", None),
            "status": getattr(report, "status", None),
            "image": jpg.name if image else None,
            "metadata": getattr(report, "metadata", {}) or {},
        }
        jso.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        logger.info("[store] {} -> {}", report_type, jso)
        return str(jso)
