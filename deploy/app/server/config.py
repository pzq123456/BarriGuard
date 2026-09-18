"""配置加载: server/config.yaml(相机/算法启用/标定文件三层) -> dict。

返回 {"server", "cameras": [{id, name, rtsp_url, algos: {算法名: 标定}}]}。
本模块只管拓扑 (相机列表、启用开关、标定路径解析)；标定内容校验归各算法包
(经 registry 分发)。
标定路径相对包根 (deploy/app) 解析，不存在则相对 config 文件所在目录再试。

配置定位（deploy 生产世界）：load_runtime() 依次取
  env BARRIGUARD_CONFIG -> ROOT/config.yaml -> ROOT.parent/config.yaml。
legacy load() 仍读 server/config.yaml（DEFAULT_PATH），行为保持不变。
"""
import os
from pathlib import Path

import yaml

from . import registry
from .config_validate import ConfigError, format_manifest, parse_runtime

DEFAULT_PATH = Path(__file__).parent / "config.yaml"
# 生产世界包根 = deploy/app；标定路径与配置定位都相对它。
ROOT = Path(__file__).resolve().parent.parent

# deploy 生产配置定位（load_runtime 默认）；legacy load() 仍读 DEFAULT_PATH。
DEFAULT_RUNTIME_PATH = ROOT.parent / "config.yaml"


def _config_path():
    """按 BARRIGUARD_CONFIG -> ROOT/config.yaml -> ROOT.parent/config.yaml 定位。"""
    env = os.environ.get("BARRIGUARD_CONFIG")
    if env:
        return Path(env)
    for candidate in (ROOT / "config.yaml", ROOT.parent / "config.yaml"):
        if candidate.is_file():
            return candidate
    return DEFAULT_RUNTIME_PATH


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


def load_runtime(path=None):
    """读 deploy/config.yaml -> contracts.RuntimeConfig（Wave 1 严格入口）。

    与 legacy load() 并存、互不影响：未知键 / 缺必填 / enabled 但缺 calibration /
    时间无法解析 / callback url 非法 -> 抛 ConfigError（启动失败），不回落默认值。
    解析成功后按 相机->算法->schedule->calibration->status 打印清单；
    status=calibration_pending 照常加载并打印，不禁用。

    路径解析顺序（未显式传 path 时）：env BARRIGUARD_CONFIG ->
    ROOT/config.yaml -> ROOT.parent/config.yaml（见 _config_path）。
    """
    fp = Path(path or _config_path())
    if not fp.is_file():
        raise ConfigError(f"运行时配置不存在: {fp}")
    data = yaml.safe_load(fp.read_text(encoding="utf-8"))
    cfg = parse_runtime(data, base_dir=fp.parent, source=str(fp))
    print(format_manifest(cfg, source=str(fp)))
    return cfg
