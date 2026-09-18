"""算法插件接口: server 只认 Algorithm 协议, 不关心内部实现。

算法每帧吃视频帧, 只出三样东西 (其余都不用管):
  events  报警 (统一 schema, 转发模块以后直接消费它)
  annots  渲染标注 (框/点, server 按 level 通用配色画出)
  debug   调试图层 {名字: 灰度/彩色图} (如 roi 掩膜, 按需加, server 原样 serving)

约定的窄接口 (专家方案):
  cadence 按帧 (per_frame, 如 water_gap) 或按周期 (periodic, 如 night_lamp 预留)。
  状态 (support 累计 / night_state 等) 是算法实例属性, 由 CameraWorker 持有生命周期。
算法包 import 本模块构造返回对象 (正常的插件方向依赖)。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable


@dataclass
class Event:
    camera_id: str
    algo: str
    ts: float
    kind: str            # "alarm" | "suspected"
    payload: dict = field(default_factory=dict)
    evidence_path: str | None = None


@dataclass
class Annotation:
    kind: str            # "box" | "point"
    box: tuple = ()      # box 用 (x0,y0,x1,y1), point 用 (x,y)
    label: str = ""
    level: str = "info"  # "alarm" | "suspected" | "info" (server 通用配色)


@dataclass
class Report:
    """算法产出的“成品”（图片/文档级），与 Event/Annotation 分离。

    additive：旧算法完全不产生 Report 也能正常工作（reports 默认空表）。
    Runtime 的 Reporter 消费它（内存队列 -> callback），算法层不碰 HTTP/磁盘。
    """
    camera: str
    algorithm: str
    report_type: str              # 如 "night_heatmap" / "gap_overlay"
    created_at: str               # ISO-8601（含时区）
    image_jpeg: bytes | None = None
    metadata: dict = field(default_factory=dict)
    status: str = "ok"            # ok | degraded | rejected | failed


@dataclass
class AlgoResult:
    events: List[Event]
    annots: List[Annotation]
    debug: Dict[str, object] = field(default_factory=dict)
    reports: List[Report] = field(default_factory=list)   # additive，默认空


@runtime_checkable
class Algorithm(Protocol):
    name: str
    cadence: str         # "per_frame" | "periodic"

    def step(self, frame, now: float) -> AlgoResult:
        ...

    def reset(self) -> None:
        ...
