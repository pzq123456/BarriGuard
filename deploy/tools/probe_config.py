"""Phase-1 config consumption probe (read-only).

Purpose: produce evidence for *which* config values actually influence runtime
behaviour, without changing any parsing/validation logic.

Two observations are recorded:

1. Path resolution -- every call to the config-discovery / calibration-resolver
   entry points, with the candidates tried and the file finally chosen.
2. Field / dict-key consumption -- dynamic tracking of attribute reads on the
   frozen ``server.contracts`` dataclasses and of item reads on config sub-dicts
   (``night_gate``) and on algorithm calibration dicts returned by the registry.

Install from a driver, run the real code paths, then ``dump()``.
Everything is inert unless ``install()`` is called.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PAUSED = False
_STATE = {
    "fields_used": set(),        # "NightLampSpec.night_gate"
    "field_callers": {},         # "NightLampSpec.night_gate" -> set of callers
    "dict_keys_used": {},        # tag -> set of keys
    "paths": [],                 # path-resolution events
}

# Reads coming from dataclass machinery (repr/eq/hash) or copy are structural,
# not semantic; they must not count as "field consumed".
_STRUCTURAL_CALLERS = frozenset({
    "__repr__", "__str__", "__eq__", "__ne__", "__hash__", "__reduce__",
    "__reduce_ex__", "__getstate__", "__setstate__", "__deepcopy__",
    "__copy__", "_asdict_inner", "asdict", "astuple", "fields", "replace",
    "_plain", "format_effective",
})


def record_field(name: str, caller: str = "") -> None:
    if _PAUSED:
        return
    with _LOCK:
        _STATE["fields_used"].add(name)
        _STATE["field_callers"].setdefault(name, set()).add(caller or "?")


def record_key(tag: str, key) -> None:
    if _PAUSED:
        return
    with _LOCK:
        _STATE["dict_keys_used"].setdefault(tag, set()).add(str(key))


def record_path(site: str, resolver: str, resolved, candidates=None,
                note: str = "") -> None:
    with _LOCK:
        _STATE["paths"].append({
            "site": site, "resolver": resolver,
            "resolved": None if resolved is None else str(resolved),
            "candidates": None if candidates is None else [str(c) for c in candidates],
            "note": note,
        })


def _caller_name() -> str:
    try:
        return sys._getframe(2).f_code.co_name
    except Exception:
        return ""


def _patch_dataclass(cls, tag: str) -> None:
    field_names = frozenset(f.name for f in dataclasses.fields(cls))
    original = cls.__getattribute__

    def __getattribute__(self, name):
        if name in field_names:
            caller = _caller_name()
            if caller not in _STRUCTURAL_CALLERS:
                record_field(f"{tag}.{name}", caller)
        return original(self, name)

    cls.__getattribute__ = __getattribute__


def patch_contracts() -> None:
    """Patch every contract dataclass so field reads are recorded."""
    from server import contracts as c

    for attr in c.__all__:
        obj = getattr(c, attr, None)
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            _patch_dataclass(obj, obj.__name__)


class TrackedDict(dict):
    """A plain dict that records which keys are read."""

    def __init__(self, *args, _tag="<dict>", **kwargs):
        super().__init__(*args, **kwargs)
        self._tag = _tag

    def __getitem__(self, key):
        record_key(self._tag, key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        record_key(self._tag, key)
        return super().get(key, default)

    def __contains__(self, key):
        record_key(self._tag, key)
        return super().__contains__(key)


def wrap_dict(d, tag):
    """Recursively wrap a mapping (and nested mappings) with TrackedDict."""
    if not isinstance(d, dict) or isinstance(d, TrackedDict):
        return d
    out = {}
    for k, v in d.items():
        out[k] = wrap_dict(v, tag) if isinstance(v, dict) else v
    return TrackedDict(out, _tag=tag)


def wrap_config_dicts(cfg) -> None:
    """Replace every dict field reachable from a config object with a TrackedDict.

    Recording is paused while traversing so the probe's own reads do not count.
    """
    global _PAUSED
    with _LOCK:
        _PAUSED = True
    try:
        _wrap_dict_fields(cfg, type(cfg).__name__, set())
    finally:
        _PAUSED = False


def _wrap_dict_fields(obj, tag, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        return
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        field_tag = f"{tag}.{f.name}"
        if isinstance(value, dict) and not isinstance(value, TrackedDict):
            setattr(obj, f.name, wrap_dict(value, field_tag))
        else:
            _wrap_dict_fields(value, field_tag, seen)


def patch_resolvers() -> None:
    """Wrap the four resolution entry points; call the originals unchanged."""
    import server.config as config
    import server.config_validate as cv
    import server.registry as registry
    import server.worker as worker

    original_config_path = config._config_path

    def _config_path():
        env = os.environ.get("BARRIGUARD_CONFIG")
        candidates = []
        if env:
            candidates.append(env)
        candidates += [config.ROOT / "config.yaml",
                       config.ROOT.parent / "config.yaml"]
        try:
            resolved = original_config_path()
            note = "env" if env else "fallback"
        except Exception as exc:  # pragma: no cover
            resolved, note = None, f"{type(exc).__name__}: {exc}"
        record_path("server.config._config_path", "config discovery",
                    resolved, candidates, note)
        return resolved

    config._config_path = _config_path

    original_load_runtime = config.load_runtime

    def load_runtime(path=None):
        fp = Path(path) if path else _config_path()
        record_path("server.config.load_runtime", "runtime config file",
                    fp, [fp], "explicit" if path else "discovery")
        return original_load_runtime(path)

    config.load_runtime = load_runtime

    original_resolve_calib = cv.resolve_resource

    def _resolve_calibration(raw, base_dir, where):
        p = Path(raw)
        candidates = [p] if p.is_absolute() else [cv.ROOT / p, Path(base_dir) / p]
        try:
            out = original_resolve_calib(raw, base_dir, where)
            record_path("server.config_validate.resolve_resource",
                        "resource", out, candidates, where)
            return out
        except Exception as exc:
            record_path("server.config_validate.resolve_resource",
                        "resource", None, candidates,
                        f"{where} -> {type(exc).__name__}: {exc}")
            raise

    cv.resolve_resource = _resolve_calibration

    original_load_calib = registry.load_calib

    def load_calib(name, fp):
        cal = original_load_calib(name, fp)
        record_path("server.registry.load_calib", f"calibration[{name}]",
                    fp, [fp], "loaded")
        return wrap_dict(cal, f"calib[{name}]")

    registry.load_calib = load_calib

    original_worker_load = worker._load_calibration

    def _load_calibration(name, spec):
        cal = getattr(spec, "calibration", "")
        record_path("server.worker._load_calibration", f"load[{name}]",
                    cal, [cal], f"schedule={getattr(spec, 'schedule', '')}")
        return original_worker_load(name, spec)

    worker._load_calibration = _load_calibration


def install() -> None:
    patch_contracts()
    patch_resolvers()
    import atexit
    atexit.register(dump)


def default_out() -> Path:
    env = os.environ.get("BARRIGUARD_PROBE_OUT")
    if env:
        return Path(env)
    base = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
    return base / "opencode" / "config_field_usage_report.json"


def _declared_fields():
    from server import contracts as c

    out = {}
    for attr in c.__all__:
        obj = getattr(c, attr, None)
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            out[obj.__name__] = [f.name for f in dataclasses.fields(obj)]
    return out


def snapshot() -> dict:
    declared = _declared_fields()
    declared_q = {f"{cls}.{f}" for cls, fs in declared.items() for f in fs}
    with _LOCK:
        used = set(_STATE["fields_used"])
        return {
            "fields_used": sorted(used),
            "field_callers": {k: sorted(v) for k, v in _STATE["field_callers"].items()},
            "fields_declared": declared,
            "fields_unread": sorted(declared_q - used),
            "dict_keys_used": {k: sorted(v) for k, v in _STATE["dict_keys_used"].items()},
            "paths": list(_STATE["paths"]),
        }


def dump(out=None) -> Path:
    out = Path(out) if out else default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
