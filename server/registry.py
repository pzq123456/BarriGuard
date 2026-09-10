"""算法注册表: server 构造/校验算法的唯一入口。

server 不散 import 各算法包 (当前只有 water_barrier；night_lamp 待其
periodic 调度就绪后在此注册)。标定内容校验归各算法包的 load_calibration，
server 只负责文件路径解析 (见 config._resolve)。
"""
from water_barrier import WaterGapAlgorithm, load_calibration

LOADERS = {"water_gap": load_calibration}
BUILDERS = {"water_gap": WaterGapAlgorithm}

# 已知但尚未接入 server 的算法：启用时给明确指引，不报"未知算法"。
DEFERRED = {"night_lamp": "仍走 night_lamp/main.py 独立运行 (periodic 调度未接入)"}


def load_calib(name, fp):
    try:
        loader = LOADERS[name]
    except KeyError:
        if name in DEFERRED:
            raise RuntimeError(f"{name} 尚未接入server ({DEFERRED[name]})")
        raise RuntimeError(f"未知算法 {name}")
    return loader(fp)


def create(name, frame_shape, calib, camera_id=""):
    try:
        cls = BUILDERS[name]
    except KeyError:
        raise RuntimeError(f"未知算法 {name}")
    return cls(frame_shape, calib, camera_id)
