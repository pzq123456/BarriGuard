"""算法注册表: server 构造/校验算法的唯一入口（deploy 生产世界版本）。

water_barrier 走这里；夜灯算法本体（NightAdapter/NightSession）由
``server.worker`` 的工厂直接构造（纯 spec 驱动，无算法侧配置文件），
不在此注册 builder。

server 不散 import 各算法包。标定内容校验归各算法包的 load_calibration，
server 只负责文件路径解析 (见 config_validate.resolve_resource)。
"""
from water_barrier import WaterGapAlgorithm, load_calibration

LOADERS = {"water_gap": load_calibration}
# night_lamp 由 worker 的 NightAdapter 工厂直接构造（见 worker._default_night_adapter）
BUILDERS = {"water_gap": WaterGapAlgorithm}


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
