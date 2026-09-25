"""Wave 1 Agent F 验收集: tests/acceptance/*.

每个子模块暴露 ``checks(ctx) -> list[(name, callable)]``。
``run_all(ctx)`` 汇总执行，返回 ``replay_harness.Result`` 列表。
pytest 亦可将各 ``test_*`` 函数作为独立用例发现。
"""
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent.parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import replay_harness as H  # noqa: E402

_SUBMODULES = (
    "test_source",
    "test_day",
    "test_contract_surface",
    "test_night_lifecycle",
    "test_failure_isolation",
    "test_runtime_isolation",
    "test_overnight_burst",
    "test_night_evidence",
    "test_reliability",
)


def run_all(ctx):
    import importlib

    items = []
    for name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        items += mod.checks(ctx)
    return H.run_checks(items)
