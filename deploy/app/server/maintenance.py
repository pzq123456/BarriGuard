"""定时清理：删除落盘目录中超过保留期的文件。

落盘只保留“必要的交换格式”（JSON / 夜间缓存），本模块按 ``retention_hours``
周期扫描 root 并删除过期文件，防止副本无限增长。任何异常只记日志，不冒泡。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger

_SWEEP_INTERVAL_S = 3600.0
_MIN_INTERVAL_S = 60.0
# 日志目录不参与 retention 清理：日志是排障底料，不能被 48h TTL 顺手删掉。
DEFAULT_SKIP_NAMES = ("logs",)


class Sweeper:
    """后台线程：每 ``interval_s`` 扫一次，删除 mtime 早于 TTL 的文件。"""

    def __init__(self, roots, retention_hours, interval_s=_SWEEP_INTERVAL_S,
                 skip_names=DEFAULT_SKIP_NAMES):
        seen: list[str] = []
        for r in roots:
            s = str(r or "").strip()
            if s and s not in seen:
                seen.append(s)
        self._roots = [Path(s) for s in seen]
        self._skip = {str(n) for n in (skip_names or ())}
        self._ttl_s = max(float(retention_hours), 0.0) * 3600.0
        self._interval_s = max(float(interval_s), _MIN_INTERVAL_S)
        self._stop = threading.Event()
        self._thread = None

    @property
    def enabled(self) -> bool:
        return self._ttl_s > 0.0 and bool(self._roots)

    def start(self) -> None:
        if not self.enabled:
            return
        self.sweep()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sweeper")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.sweep()

    def sweep(self) -> int:
        """删除过期文件，返回清理数量。"""
        cutoff = time.time() - self._ttl_s
        removed = 0
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                try:
                    if self._skip.intersection(path.parts):
                        continue
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        if removed:
            logger.info("[sweeper] 清理 {} 个超过 {}h 的文件", removed,
                        self._ttl_s / 3600.0)
        return removed
