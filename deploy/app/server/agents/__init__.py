"""Wave 1 运行时 agent 封装（白昼水马等）。

只做薄封装：透传算法、内存出图，不改算法、不落盘、不做告警节流。
"""
from .day import DayRunner

__all__ = ["DayRunner"]
