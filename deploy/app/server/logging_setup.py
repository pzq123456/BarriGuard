"""运行期日志：控制台 + 落盘文件（部署日志）。

服务器上原来只写 stderr（docker logs），进程/容器一重启就丢，夜间与白昼的
算法行为无法回溯。这里加一个按天切分的文件 sink（默认 ``<output_dir>/logs``），
并把生效配置写进日志首行，方便"采集一版日志"直接定位问题。

用法：``configure(level, output_dir, log_dir, banner=...)``；同参重复调用为
no-op，``__main__`` 与 ``app`` 两条入口都调也不会挂两份 sink。
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

LOG_SUBDIR = "logs"
LOG_FILE = "barriguard_{time:YYYY-MM-DD}.log"
FILE_LEVEL = "INFO"
CONSOLE_FALLBACK = "WARNING"
ROTATION = "20 MB"
RETENTION = "14 days"
FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{line} | {message}"

_configured = None  # 最近一次 (level, log_dir)；重复调用跳过，避免多份 sink


def log_dir_for(output_dir=None, log_dir=None) -> str:
    """显式 log_dir 优先；否则 <output_dir>/logs；两者皆空则只写控制台。"""
    override = str(log_dir or "").strip()
    if override:
        return override
    out = str(output_dir or "").strip()
    return str(Path(out) / LOG_SUBDIR) if out else ""


def configure(level="warning", output_dir=None, log_dir=None, banner=None):
    """装 stderr + 文件两个 sink；返回日志文件路径模板（未落盘则 None）。"""
    global _configured
    target = log_dir_for(output_dir, log_dir)
    key = (str(level or "").lower(), target)
    if key == _configured:
        return None
    _configured = key

    logger.remove()
    logger.add(sys.stderr, level=str(level or CONSOLE_FALLBACK).upper(),
               enqueue=True, backtrace=False, diagnose=False)

    path = None
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        path = str(Path(target) / LOG_FILE)
        logger.add(path, level=FILE_LEVEL, rotation=ROTATION,
                   retention=RETENTION, encoding="utf-8", enqueue=True,
                   backtrace=True, diagnose=False, format=FORMAT)
    if banner:
        logger.info(banner)
    return path


def startup_banner(cfg, source="") -> str:
    """生效配置横幅（写进日志，便于把一次部署的日志与配置对齐）。"""
    from .config_validate import format_effective, format_manifest
    return "\n".join(["BarriGuard 启动", format_manifest(cfg, source=source),
                      format_effective(cfg, source=source)])
