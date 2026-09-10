"""配置加载: server/config.yaml(相机/算法启用/标定文件三层) -> dict。

返回 {"server", "cameras": [{id, name, rtsp_url, algos: {算法名: 标定}}]}。
本模块只管拓扑 (相机列表、启用开关、标定路径解析)；标定内容校验归各算法包
(经 registry 分发)。
标定路径相对仓库根解析，不存在则相对 config 文件所在目录再试。
"""
from pathlib import Path

import yaml

from . import registry

DEFAULT_PATH = Path(__file__).parent / "config.yaml"
ROOT = Path(__file__).resolve().parent.parent


def _resolve(path, base):
    p = Path(path)
    if p.is_absolute():
        return p
    for parent in (ROOT, base):
        c = parent / p
        if c.is_file():
            return c
    raise RuntimeError(f"标定文件不存在: {path}")


def load(path=None):
    """读部署配置, 返回 {"server", "cameras"}。无启用相机会抛错。"""
    fp = Path(path or DEFAULT_PATH)
    d = yaml.safe_load(fp.read_text(encoding="utf-8"))
    base = fp.parent
    cams = []
    for c in d.get("cameras", []):
        if not c.get("enabled", True):
            continue
        algos = {}
        for name, a in (c.get("algorithms") or {}).items():
            if not (a or {}).get("enabled", False):
                continue
            algos[name] = registry.load_calib(name, _resolve(a["calibration"], base))
        cams.append({"id": str(c["id"]), "name": c.get("name") or str(c["id"]),
                     "rtsp_url": c["rtsp_url"], "algos": algos})
    if not cams:
        raise RuntimeError("没有启用任何相机")
    return {"server": d["server"], "cameras": cams}
