"""配置加载: deploy/config.yaml -> contracts.RuntimeConfig（严格校验）。

本模块只管拓扑 (相机列表、启用开关、标定路径解析)；标定内容校验归各算法包
(经 registry 分发)。标定路径相对包根 (deploy/app) 解析，不存在则相对 config
文件所在目录再试。

配置定位：load_runtime() 依次取
  env BARRIGUARD_CONFIG -> ROOT/config.yaml -> ROOT.parent/config.yaml。
"""
import os
from pathlib import Path

import yaml

from .config_validate import (
    ConfigError, format_effective, format_manifest, parse_runtime,
)

# 生产世界包根 = deploy/app；标定路径与配置定位都相对它。
ROOT = Path(__file__).resolve().parent.parent

# deploy 生产配置定位（load_runtime 默认）。
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


def load_runtime(path=None):
    """读 deploy/config.yaml -> contracts.RuntimeConfig（Wave 1 严格入口）。

    未知键 / 缺必填 / enabled 但缺 calibration / 时间无法解析 / callback url 非法
    -> 抛 ConfigError（启动失败），不回落默认值。
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
    print(format_effective(cfg, source=str(fp)))
    return cfg
