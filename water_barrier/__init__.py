"""Water-barrier gap alarm algorithm (self-contained algorithm package).

server/ 只认 server.algo.Algorithm 协议, 经注册表按名构造本包算法：

    from water_barrier import WaterGapAlgorithm, load_calibration
    calib = load_calibration(path)          # 标定校验归本包
    algo = WaterGapAlgorithm(shape, calib)  # 每流一份, 状态实例持有
    res = algo.step(frame_bgr, now)         # 事件 + 标注 + 调试图层

分层：signal（单帧信号链，纯函数）/ track（support 累计 + 时序确认）/
pipeline（标定 + 装配 + server 协议）。离线验证走 server 入口：
python -m server --offscreen <img> / python -m server.regression。
"""

from .pipeline import WaterGapAlgorithm, load_calibration

__all__ = ["WaterGapAlgorithm", "load_calibration"]
