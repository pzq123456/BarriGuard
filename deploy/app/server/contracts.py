"""Wave 0 冻结契约：运行时 / 调度 / 相机-算法绑定 / 夜灯 session 配置。

本模块只定义**数据结构**，不含加载与校验逻辑（由 Wave 1 的 config loader 实现）。
边界原则：
  部署配置（deploy/config.yaml）  -> 本模块的数据类（跑什么/何时跑/输出给谁）
  业务策略（confidence/min_interval/hold/reconfirm）-> 本模块
  算法数据资产（水马 ROI/road 几何标定）-> 留在算法包，配置只给路径引用

注意：
  - Report 定义在 server/algo.py（属于 Algorithm 产物契约），此处 import 复用。
  - schedule 字符串只用 "day" / "night"；Runtime 按 config 决定启停，
    禁止在代码里写死 "NIGHT 就关水马" 之类的业务逻辑。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .algo import Report  # noqa: F401  (re-export, 供 Runtime/Reporter 使用)

__all__ = [
    "Report", "ScheduleState", "CalibrationStatus", "AlignmentStatus",
    "ScheduleSpec", "CallbackSpec", "WaterGapSpec", "DetectSpec", "TrackSpec",
    "SamplingSpec",
    "MemorySpec", "BaselineSpec", "PeriodicitySpec", "AlignmentSpec",
    "OvernightSpec", "NightLampSpec", "AlgorithmBinding", "CameraSpec",
    "RuntimeConfig",
]


class ScheduleState(str, Enum):
    """Runtime 全局状态机。REPORT 不再是独立运行模式：它是 FINALIZING 的输出事件。"""
    OFF = "off"
    DAY = "day"            # 相机按各自 algo.schedule 活跃
    NIGHT = "night"
    FROZEN = "frozen"      # 夜间采样停止，保留 NightSession 等待 07:00
    FINALIZING = "finalizing"


class CalibrationStatus(str, Enum):
    READY = "ready"
    PENDING = "calibration_pending"   # 打印但不禁用；不得静默关闭


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    DEGRADED = "degraded"              # 低 cc：可发布但显式标记
    REJECTED = "rejected"             # 无法对齐：不发布“看似正常”的图
    BASE_UNAVAILABLE = "base_frame_unavailable"   # 07:00 拉不到当前帧
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ScheduleSpec:
    timezone: str = "Asia/Shanghai"
    day_start: str = "07:00"
    day_end: str = "17:00"
    night_start: str = "20:00"
    night_end: str = "03:00"
    report_at: str = "07:00"


@dataclass
class CallbackSpec:
    url: str = ""
    timeout_s: float = 8.0
    queue_size: int = 64              # 必须有界，防止 callback 挂起导致内存泄漏
    image_field: str = "image_base64"
    enabled: bool = True
    retries: int = 0                  # 失败后额外重试次数（0 = 不重试）


@dataclass
class DetectSpec:
    """水马单帧检测参数（原标定文件 detect 段，上移为运行期选项）。"""
    trim: float = 0.04
    conf_th: float = 0.30
    support_th: float = 0.3
    min_box_width: int = 30
    floor_max: float = 0.24
    core_exit: float = 0.06
    min_width: int = 40
    core_min_width: int = 30


@dataclass
class TrackSpec:
    """水马时序跟踪参数（原标定文件 track 段非策略键，上移为运行期选项）。

    rer_threshold/alarm_hold_s/reconfirm_s 不在此处：它们是 WaterGapSpec 的
    策略字段（confidence/alarm_hold_s/reconfirm_s），注入时同名写入 track。
    """
    decay_rate: float = 0.5
    track_stale_s: float = 8.0
    match_iou: float = 0.25
    match_center_ratio: float = 0.5
    median_window: int = 5
    intact_reset_s: float = 5.0
    rer_purity: float = 0.70
    patch_size: int = 8


@dataclass
class WaterGapSpec:
    """水马：算法逻辑不变，只配业务策略与出图节奏。"""
    calibration: str = ""
    schedule: str = "day"             # 由 config 决定，不写死
    status: CalibrationStatus = CalibrationStatus.READY  # manifest-only, no runtime branch

    report_interval_s: int = 3600     # 每小时一张缺口图
    alarm_min_interval_s: int = 3600  # 最小告警间隔（与 report 独立）
    confidence: float = 0.25          # RER 门（track.rer_threshold）

    # 状态机（注入算法，算法内部不硬编码）
    alarm_hold_s: float = 10.0
    reconfirm_s: float = 5.0

    detect: DetectSpec = field(default_factory=DetectSpec)
    track: TrackSpec = field(default_factory=TrackSpec)


@dataclass
class SamplingSpec:
    interval_ms: int = 160            # Phase 1 起点（候选值，非真理）


@dataclass
class MemorySpec:
    max_candidates: int = 200         # 跟踪容量上限（内存护栏），溢出记入 metadata，不影响 status
    series_cap: Optional[int] = None  # None = 全序列（受 max_candidates 约束）；
    #                                   600 仅为 benchmark 候选，禁止作 production 默认


@dataclass
class BaselineSpec:
    frames: int = 60
    warmup_s: int = 600


@dataclass
class PeriodicitySpec:
    period_step: int = 2
    lag_lo_s: float = 0.3
    lag_hi_s: float = 7.0
    peak_floor: float = 0.2
    min_peaks: int = 3
    gap_tol: int = 1
    onset_min: int = 50


@dataclass
class AlignmentSpec:
    min_cc: float = 0.5               # 低于此 -> degraded（或 rejected，按 on_low_cc）
    on_low_cc: AlignmentStatus = AlignmentStatus.DEGRADED


@dataclass
class OvernightSpec:
    """夜间 burst 调度：每小时一段连续 burst，空间累计跨 burst 保留。

    enabled=False 时退回旧的“整夜单一 NightSession”行为。

    anchor:
      * ``"stream"``（默认/旧行为）：burst 从「首帧」起用单调钟推进。
      * ``"wall"``：由墙钟整点分桶驱动——每小时前 ``burst_seconds`` 观测，
        桶末由独立的 tick（非 on_frame）触发结算并出图；断流也照常出图。
    """
    enabled: bool = False
    cadence_minutes: int = 60         # burst 间隔
    burst_seconds: int = 120          # 每段连续观测时长
    anchor: str = "stream"            # stream | wall


@dataclass
class NightLampSpec:
    """夜灯：以 nightly_map 为目标算法，整夜在线累积，07:00 finalize。"""
    schedule: str = "night"
    status: CalibrationStatus = CalibrationStatus.READY  # manifest-only, no runtime branch

    night_gate: dict = field(default_factory=dict)   # enter/exit/persistence
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    periodicity: PeriodicitySpec = field(default_factory=PeriodicitySpec)
    alignment: AlignmentSpec = field(default_factory=AlignmentSpec)
    overnight: OvernightSpec = field(default_factory=OvernightSpec)


@dataclass
class AlgorithmBinding:
    schedule: str                     # "day" | "night"
    spec: object                      # WaterGapSpec | NightLampSpec


@dataclass
class CameraSpec:
    id: str
    name: str
    rtsp_url: str
    enabled: bool = True
    algorithms: dict = field(default_factory=dict)   # name -> AlgorithmBinding


@dataclass
class RuntimeConfig:
    schedule: ScheduleSpec = field(default_factory=ScheduleSpec)
    callback: CallbackSpec = field(default_factory=CallbackSpec)
    cameras: list = field(default_factory=list)      # list[CameraSpec]
    server: dict = field(default_factory=dict)       # host/port/jpeg_quality/...
    memory_budget_mb_per_camera: int = 500           # manifest-only, not consumed at runtime
    output_dir: str = ""                             # 空 = 不落盘；host-mounted root
    alarm_dir: str = ""                              # 告警帧落盘根；空则回退 output_dir
    persist_images: bool = True                      # False = 不落图片，仅保留交换格式(JSON)
    retention_hours: float = 0.0                     # >0 时定时清理落盘目录中超过该时长的文件
