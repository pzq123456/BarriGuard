"""算法注册表: server 构造/校验算法的唯一入口。

server 不散 import 各算法包。标定内容校验归各算法包的 load_calibration，
server 只负责文件路径解析 (见 config._resolve)。
"""
from night_lamp.server_plugin import NightLampAlgorithm
from night_lamp.server_plugin import load_calibration as load_night_lamp
from water_barrier import WaterGapAlgorithm, load_calibration

LOADERS = {"water_gap": load_calibration, "night_lamp": load_night_lamp}
BUILDERS = {"water_gap": WaterGapAlgorithm, "night_lamp": NightLampAlgorithm}


def load_calib(name, fp):
    try:
        loader = LOADERS[name]
    except KeyError:
        raise RuntimeError(f"未知算法 {name}")
    return loader(fp)


def create(name, frame_shape, calib, camera_id=""):
    try:
        cls = BUILDERS[name]
    except KeyError:
        raise RuntimeError(f"未知算法 {name}")
    return cls(frame_shape, calib, camera_id)
