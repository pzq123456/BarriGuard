"""Wave 1 运行时配置严格校验（归 A 所有）。

只服务 ``server.config.load_runtime()``：把 ``deploy/config.yaml`` 解析成
``contracts.RuntimeConfig``，任何**未知键 / 缺必填 / 非法值**都立即抛
``ConfigError``，绝不回落默认值、绝不静默跳过。

边界：
  - 本模块只校验“跑什么 / 何时跑 / 输出给谁”的拓扑与显式字段；
    算法标定文件的内容仍由各算法包负责（此处只解析路径并要求文件存在）。
"""
from __future__ import annotations

import typing
from dataclasses import fields as _dc_fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from .contracts import (
    AlgorithmBinding,
    AlignmentSpec,
    BaselineSpec,
    CalibrationStatus,
    CallbackSpec,
    CameraSpec,
    MemorySpec,
    NightLampSpec,
    OvernightSpec,
    PeriodicitySpec,
    RuntimeConfig,
    SamplingSpec,
    ScheduleSpec,
    WaterGapSpec,
)

ROOT = Path(__file__).resolve().parent.parent

__all__ = ["ConfigError", "parse_runtime", "format_manifest"]


class ConfigError(RuntimeError):
    """运行时配置非法：启动即失败。"""


# --- schema：允许键 -----------------------------------------------------------------
TOP_KEYS = {"version", "runtime", "schedule", "callback", "cameras"}
TOP_REQUIRED = ("version", "runtime", "schedule", "callback", "cameras")

RUNTIME_KEYS = {
    "host", "port", "jpeg_quality", "log_level", "memory_budget_mb_per_camera",
    "output_dir",
}
RUNTIME_REQUIRED = ("host", "port")

SCHEDULE_KEYS = {
    "timezone", "day_start", "day_end", "night_start", "night_end", "report_at",
}
SCHEDULE_REQUIRED = tuple(SCHEDULE_KEYS)
TIME_KEYS = ("day_start", "day_end", "night_start", "night_end", "report_at")

CALLBACK_KEYS = {"enabled", "url", "timeout_s", "queue_size", "image_field"}
CALLBACK_REQUIRED = ("enabled", "url")

CAMERA_KEYS = {"id", "name", "rtsp_url", "enabled", "algorithms"}
CAMERA_REQUIRED = ("id", "rtsp_url")

BINDING_KEYS = {"enabled", "schedule", "calibration", "status"}
ALLOWED_SCHEDULES = {"day", "night"}

# 算法名 -> 绑定 spec 的数据类（新增算法在此登记；不在代码里散落算法名判断）
ALGO_SPECS = {"water_gap": WaterGapSpec, "night_lamp": NightLampSpec}
# night_lamp 的嵌套 spec 键 -> 数据类
NESTED_SPECS = {
    "sampling": SamplingSpec,
    "memory": MemorySpec,
    "baseline": BaselineSpec,
    "periodicity": PeriodicitySpec,
    "alignment": AlignmentSpec,
    "overnight": OvernightSpec,
}
# NightLampSpec.night_gate 是裸 dict（contracts 未定数据类），显式列出允许键
NIGHT_GATE_KEYS = {"enter_threshold", "exit_threshold", "persistence"}


# --- 基础校验助手 -------------------------------------------------------------------
def _reject_unknown(m, allowed, where):
    if not isinstance(m, dict):
        raise ConfigError(f"{where} 必须是映射(mapping)，实际 {type(m).__name__}")
    unknown = sorted(str(k) for k in m if k not in allowed)
    if unknown:
        raise ConfigError(f"{where} 含有未知键: {', '.join(unknown)}")


def _require(m, keys, where):
    if not isinstance(m, dict):
        raise ConfigError(f"{where} 必须是映射(mapping)，实际 {type(m).__name__}")
    missing = [k for k in keys if k not in m]
    if missing:
        raise ConfigError(f"{where} 缺少必填键: {', '.join(missing)}")


def _as_str(v, where, allow_empty=False):
    if not isinstance(v, str) or (not allow_empty and not v.strip()):
        raise ConfigError(f"{where} 必须是非空字符串")
    return v


def _as_bool(v, where):
    if not isinstance(v, bool):
        raise ConfigError(f"{where} 必须是布尔值 true/false，实际 {v!r}")
    return v


def _as_int(v, where, minimum=None):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where} 必须是整数，实际 {v!r}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where} 必须 >= {minimum}，实际 {v}")
    return v


def _as_float(v, where, minimum=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where} 必须是数值，实际 {v!r}")
    val = float(v)
    if minimum is not None and val < minimum:
        raise ConfigError(f"{where} 必须 >= {minimum}，实际 {val}")
    return val


def _parse_time(v, where):
    s = _as_str(v, where)
    try:
        datetime.strptime(s, "%H:%M")
    except ValueError:
        raise ConfigError(f"{where} 时间无法解析: {s!r}（应为 HH:MM）")
    return s


def _parse_enum(cls, v, where):
    try:
        return cls(v)
    except ValueError:
        allowed = ", ".join(e.value for e in cls)
        raise ConfigError(f"{where} 取值非法: {v!r}（允许: {allowed}）")


def _coerce(ftype, v, where):
    """按 contracts 数据类字段类型做严格标量/枚举/可选值转换。"""
    origin = typing.get_origin(ftype)
    if origin is typing.Union:  # Optional[X]
        args = [a for a in typing.get_args(ftype) if a is not type(None)]
        if v is None:
            return None
        return _coerce(args[0], v, where)
    if ftype is bool:
        return _as_bool(v, where)
    if ftype is int:
        return _as_int(v, where)
    if ftype is float:
        return _as_float(v, where)
    if ftype is str:
        return _as_str(v, where)
    if isinstance(ftype, type) and issubclass(ftype, Enum):
        return _parse_enum(ftype, v, where)
    return v


# --- 各段解析 -----------------------------------------------------------------------
def _resolve_calibration(raw, base_dir, where):
    s = _as_str(raw, where)
    p = Path(s)
    if p.is_absolute():
        if not p.is_file():
            raise ConfigError(f"{where} 标定文件不存在: {p}")
        return str(p)
    for parent in (ROOT, base_dir):
        c = parent / p
        if c.is_file():
            return str(c)
    raise ConfigError(f"{where} 标定文件不存在: {s}（已相对仓库根与配置文件目录查找）")


def _parse_runtime(d, where):
    _reject_unknown(d, RUNTIME_KEYS, where)
    _require(d, RUNTIME_REQUIRED, where)
    host = _as_str(d["host"], f"{where}.host")
    port = _as_int(d["port"], f"{where}.port", minimum=1)
    server = {"host": host, "port": port}
    if "jpeg_quality" in d:
        q = _as_int(d["jpeg_quality"], f"{where}.jpeg_quality", minimum=1)
        if q > 100:
            raise ConfigError(f"{where}.jpeg_quality 必须在 1..100，实际 {q}")
        server["jpeg_quality"] = q
    if "log_level" in d:
        server["log_level"] = _as_str(d["log_level"], f"{where}.log_level")
    if "output_dir" in d:
        server["output_dir"] = _as_str(d["output_dir"], f"{where}.output_dir")
    mem = None
    if "memory_budget_mb_per_camera" in d:  # manifest-only, optional
        mem = _as_int(
            d["memory_budget_mb_per_camera"],
            f"{where}.memory_budget_mb_per_camera",
            minimum=1,
        )
    return server, mem


def _parse_schedule(d, where):
    _reject_unknown(d, SCHEDULE_KEYS, where)
    _require(d, SCHEDULE_REQUIRED, where)
    kwargs = {"timezone": _as_str(d["timezone"], f"{where}.timezone")}
    for k in TIME_KEYS:
        kwargs[k] = _parse_time(d[k], f"{where}.{k}")
    return ScheduleSpec(**kwargs)


def _parse_callback(d, where):
    _reject_unknown(d, CALLBACK_KEYS, where)
    _require(d, CALLBACK_REQUIRED, where)
    enabled = _as_bool(d["enabled"], f"{where}.enabled")
    url = d.get("url", "")
    if not enabled and not url:
        url = ""
    else:
        url = _as_str(url, f"{where}.url")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigError(
                f"{where}.url 非法: {url!r}（需 http(s)://host[:port]/path）"
            )
    kwargs = {"enabled": enabled, "url": url}
    if "timeout_s" in d:
        kwargs["timeout_s"] = _as_float(d["timeout_s"], f"{where}.timeout_s", minimum=0.0)
    if "queue_size" in d:
        kwargs["queue_size"] = _as_int(d["queue_size"], f"{where}.queue_size", minimum=1)
    if "image_field" in d:
        kwargs["image_field"] = _as_str(d["image_field"], f"{where}.image_field")
    return CallbackSpec(**kwargs)


def _parse_night_gate(d, where):
    _reject_unknown(d, NIGHT_GATE_KEYS, where)
    out = {}
    if "enter_threshold" in d:
        out["enter_threshold"] = _as_float(d["enter_threshold"], f"{where}.enter_threshold")
    if "exit_threshold" in d:
        out["exit_threshold"] = _as_float(d["exit_threshold"], f"{where}.exit_threshold")
    if "persistence" in d:
        out["persistence"] = _as_int(d["persistence"], f"{where}.persistence", minimum=1)
    return out


def _parse_nested(cls, d, where):
    _reject_unknown(d, {f.name for f in _dc_fields(cls)}, where)
    kwargs = {}
    for f in _dc_fields(cls):
        if f.name in d:
            kwargs[f.name] = _coerce(f.type, d[f.name], f"{where}.{f.name}")
    return cls(**kwargs)


def _binding_allowed(name):
    return BINDING_KEYS | {f.name for f in _dc_fields(ALGO_SPECS[name])}


def _build_spec(name, b, schedule, calibration, status, where):
    cls = ALGO_SPECS[name]
    kwargs = {"calibration": calibration, "schedule": schedule, "status": status}
    for f in _dc_fields(cls):
        key = f.name
        if key in ("calibration", "schedule", "status") or key not in b:
            continue
        kwhere = f"{where}.{key}"
        if key in NESTED_SPECS:
            kwargs[key] = _parse_nested(NESTED_SPECS[key], b[key], kwhere)
        elif key == "night_gate":
            kwargs[key] = _parse_night_gate(b[key], kwhere)
        else:
            kwargs[key] = _coerce(f.type, b[key], kwhere)
    return cls(**kwargs)


def _parse_binding(name, b, where, base_dir):
    if name not in ALGO_SPECS:
        raise ConfigError(
            f"{where} 未知算法名 {name!r}（支持: {', '.join(sorted(ALGO_SPECS))}）"
        )
    _reject_unknown(b, _binding_allowed(name), where)
    _require(b, ("enabled",), where)
    enabled = _as_bool(b["enabled"], f"{where}.enabled")
    if not enabled:
        return None  # 显式禁用：跳过（status 不影响启用）
    _require(b, ("schedule", "calibration"), where)
    schedule = _as_str(b["schedule"], f"{where}.schedule")
    if schedule not in ALLOWED_SCHEDULES:
        raise ConfigError(
            f"{where}.schedule 非法: {schedule!r}（只允许 day/night）"
        )
    calibration = _resolve_calibration(b["calibration"], base_dir, f"{where}.calibration")
    status = _parse_enum(
        CalibrationStatus, b.get("status", "ready"), f"{where}.status"
    )
    spec = _build_spec(name, b, schedule, calibration, status, where)
    return AlgorithmBinding(name=name, schedule=schedule, spec=spec)


def _parse_cameras(raw, base_dir):
    if not isinstance(raw, list):
        raise ConfigError(f"cameras 必须是列表，实际 {type(raw).__name__}")
    cams = []
    seen = set()
    for i, c in enumerate(raw):
        where = f"cameras[{i}]"
        _reject_unknown(c, CAMERA_KEYS, where)
        _require(c, CAMERA_REQUIRED, where)
        cam_id = str(c["id"]).strip()
        if not cam_id:
            raise ConfigError(f"{where}.id 不能为空")
        if cam_id in seen:
            raise ConfigError(f"{where}.id 重复: {cam_id!r}")
        seen.add(cam_id)
        rtsp_url = _as_str(c["rtsp_url"], f"{where}.rtsp_url")
        enabled = _as_bool(c.get("enabled", True), f"{where}.enabled")
        if not enabled:
            continue
        algos = c.get("algorithms")
        if not isinstance(algos, dict) or not algos:
            raise ConfigError(f"{where} enabled=true 但未配置任何算法 (algorithms)")
        bindings = {}
        for name, b in algos.items():
            binding = _parse_binding(name, b, f"{where}.algorithms.{name}", base_dir)
            if binding is not None:
                bindings[name] = binding
        if not bindings:
            raise ConfigError(f"{where} enabled=true 但所有算法均被禁用")
        cams.append(
            CameraSpec(
                id=cam_id,
                name=str(c.get("name") or cam_id),
                rtsp_url=rtsp_url,
                enabled=True,
                algorithms=bindings,
            )
        )
    return cams


def parse_runtime(data, base_dir, source="<config>"):
    """严格解析 + 校验一个已 yaml.safe_load 的映射 -> RuntimeConfig。"""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: 顶层必须是映射(mapping)")
    _reject_unknown(data, TOP_KEYS, source)
    _require(data, TOP_REQUIRED, source)
    version = _as_int(data["version"], f"{source}.version")
    if version != 1:
        raise ConfigError(f"{source}.version 不支持: {version}（期望 1）")
    server, mem = _parse_runtime(data["runtime"], f"{source}.runtime")
    schedule = _parse_schedule(data["schedule"], f"{source}.schedule")
    callback = _parse_callback(data["callback"], f"{source}.callback")
    cameras = _parse_cameras(data["cameras"], Path(base_dir))
    if not cameras:
        raise ConfigError(f"{source}: 没有启用任何相机")
    kwargs = dict(
        schedule=schedule,
        callback=callback,
        cameras=cameras,
        server=server,
        output_dir=server.get("output_dir", ""),
    )
    if mem is not None:
        kwargs["memory_budget_mb_per_camera"] = mem
    return RuntimeConfig(**kwargs)


def format_manifest(cfg, source=""):
    """按 相机 -> 算法 -> schedule -> calibration -> status 打印清单。"""
    s = cfg.schedule
    c = cfg.callback
    lines = ["Runtime Configuration"]
    if source:
        lines.append(f"  source    : {source}")
    lines.append(
        f"  schedule  : tz={s.timezone} day={s.day_start}-{s.day_end} "
        f"night={s.night_start}-{s.night_end} report_at={s.report_at}"
    )
    lines.append(
        f"  callback  : enabled={c.enabled} url={c.url or '-'} "
        f"timeout={c.timeout_s}s queue={c.queue_size} field={c.image_field}"
    )
    lines.append(
        f"  server    : host={cfg.server.get('host')} port={cfg.server.get('port')} "
        f"jpeg_quality={cfg.server.get('jpeg_quality', '-')} "
        f"log_level={cfg.server.get('log_level', '-')} "
        f"mem_budget={cfg.memory_budget_mb_per_camera}MB"
    )
    lines.append(f"  cameras   : {len(cfg.cameras)} enabled")
    for cam in cfg.cameras:
        lines.append(f"    [{cam.id}] {cam.name}")
        for name, binding in cam.algorithms.items():
            spec = binding.spec
            status = spec.status.value if isinstance(spec.status, CalibrationStatus) else spec.status
            lines.append(
                f"      - algorithm={name} schedule={binding.schedule} "
                f"calibration={spec.calibration} status={status}"
            )
    return "\n".join(lines)
