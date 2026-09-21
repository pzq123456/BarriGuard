"""NightSession 状态落盘：把空间累计写成原子 npz，支持跨重启续跑。

只序列化「跨 burst / 跨进程必须保留」的东西：
    _base / _bg / _on / _peak  （空间累计）
    _n_seen / _detrend / gate  （计数与夜间资格）
时间序列（TemporalSeries / CandidateDiscovery）逐 burst 重建，刻意不落盘，
因此快照点选在 burst 边界：那一刻时间态本就该重置。

写入用「同目录临时文件 + os.replace」保证原子，进程中途被杀不会读到半文件。
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
from loguru import logger

STATE_VERSION = 1
STATE_DIR = "night_state"
ARCHIVE_DIR = "night_archive"


def night_key(ts_wall) -> str:
    """夜晚归属日期：以 12:00 为界。

    20:00 与次日 07:00 归同一个 key，跨零点不会切成两夜。
    """
    return (ts_wall - timedelta(hours=12)).date().isoformat()


class NightStateStore:
    """按 (camera_id, night_key) 读写夜间累计快照；root 为空则整体 no-op。"""

    def __init__(self, root):
        root = str(root or "").strip()
        self._root = (Path(root) / STATE_DIR) if root else None

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def path_for(self, camera_id, key):
        if not self.enabled:
            return None
        return self._root / f"{camera_id}_{key}.npz"

    def save(self, camera_id, key, state):
        path = self.path_for(camera_id, key)
        if path is None or not state:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **state)
            os.replace(tmp, path)
        except Exception:
            logger.exception("[night_state] 落盘失败 {}", path)
            self._unlink(tmp)
            return None
        return path

    def load(self, camera_id, key):
        path = self.path_for(camera_id, key)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                state = {k: z[k] for k in z.files}
        except Exception:
            logger.exception("[night_state] 读取失败 {}", path)
            return None
        if int(state.get("version", 0)) != STATE_VERSION:
            logger.warning("[night_state] 版本不符，忽略 {}", path)
            return None
        return state

    def clear(self, camera_id, key):
        self._unlink(self.path_for(camera_id, key))

    @staticmethod
    def _unlink(path):
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("[night_state] 清理失败 {}", path)


class NightArchive:
    """夜间热力图核心数据归档（L1）：按夜覆盖写，出图后不清理。

    存 ``state.npz``（on/peak/base/bg/n_seen/detrend + 派生 duty/swing）与
    ``manifest.json``；供离线复算热力图，不做时序/原始帧。
    """

    def __init__(self, root):
        root = str(root or "").strip()
        self._root = (Path(root) / ARCHIVE_DIR) if root else None

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def dir_for(self, camera_id, key):
        if not self.enabled:
            return None
        return self._root / str(key) / str(camera_id)

    def save(self, camera_id, key, state, manifest=None):
        outdir = self.dir_for(camera_id, key)
        if outdir is None or not state:
            return None
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            self._write_npz(outdir / "state.npz", self._core(state))
            if manifest is not None:
                self._write_json(outdir / "manifest.json", manifest)
        except Exception:
            logger.exception("[night_archive] 归档失败 {}", outdir)
            return None
        return str(outdir)

    @staticmethod
    def _core(state) -> dict:
        core = dict(state)
        on = np.asarray(state["on"], np.float32)
        n_seen = float(int(state["n_seen"]))
        core["duty"] = (on / n_seen) if n_seen > 0 else on
        core["swing"] = np.asarray(state["peak"], np.float32)
        return core

    @staticmethod
    def _atomic(path) -> str:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        return fd, tmp

    def _write_npz(self, path, state):
        fd, tmp = self._atomic(path)
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **state)
        os.replace(tmp, path)

    def _write_json(self, path, obj):
        fd, tmp = self._atomic(path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
