"""Water-barrier gap alarm algorithm (self-contained algorithm package).

server/ 只认 server.algo.Algorithm 协议, 经注册表按名构造本包算法：

    from water_barrier import WaterGapAlgorithm, load_calibration
    calib = load_calibration(path)          # 标定校验归本包
    algo = WaterGapAlgorithm(shape, calib)  # 每流一份, 状态实例持有
    res = algo.step(frame_bgr, now)         # 事件 + 标注 + 调试图层

Public surface: alarm.BarrierAlarm, alarm.Gap, alarm.detect_frame,
pipeline.WaterGapAlgorithm, calibration.load_calibration.
Offline validation entry: python -m water_barrier.main
"""

from .alarm import BarrierAlarm, Gap, detect_frame
from .calibration import load_calibration
from .pipeline import WaterGapAlgorithm

__all__ = ["BarrierAlarm", "Gap", "detect_frame",
           "WaterGapAlgorithm", "load_calibration"]
