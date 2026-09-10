"""配置加载：server/config.yaml -> 普通 dict。

不做对象封装：本服务只有一个消费路径，7 个 dataclass 只是把 yaml key
抄一遍（还附带翻译层 _water_gap），零行为增益。这里只抽框架真正读的键，
yaml 里其余键（检测阈值等冻结在 water_barrier.alarm）一律不碰。
"""
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "config.yaml"


def load(path=None):
    """返回 {"server", "rtsp_url", "water_gap": {"track", "road_rois", "polys"}}。"""
    d = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    wg = d["algorithms"]["water_gap"]
    region = wg.get("region", {})
    rtsp = next((c["rtsp_url"] for c in d.get("cameras", []) if c.get("enabled")), None)
    if not rtsp:
        raise RuntimeError("没有启用任何相机")
    return {
        "server": d["server"],
        "rtsp_url": rtsp,
        "water_gap": {
            "track": wg["track"],
            "road_rois": [tuple(r) for r in region["road_rois"]],
            # 缺口 ROI 多边形覆盖（归一化坐标）；缺省用 water_barrier 内冻结标定。
            "polys": ([tuple(map(tuple, p)) for p in region["polys"]]
                      if region.get("polys") else None),
        },
    }
