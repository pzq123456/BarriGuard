"""冻结签名验收: Wave 1 各生产模块必须与 tmp/wave1-interfaces.md 一致。

模块缺失 -> UNVERIFIED(依赖未落地)；模块存在但签名不符 -> FAIL(契约违规)。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "app"
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402

_GROUPS = (
    ("contract.schedule", "server.schedule"),
    ("contract.day", "server.agents.day"),
    ("contract.night_session", "night_lamp.session"),
    ("contract.night_adapter", "night_lamp.adapter"),
    ("contract.reporting", "server.reporting"),
)


def test_frozen_signatures():
    failures = []
    unverified = []
    for _name, module in _GROUPS:
        try:
            H.check_signatures(module)
        except H.Unverified as e:
            unverified.append(f"{module}: {e}")
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        raise AssertionError("; ".join(failures))
    if unverified:
        raise H.Unverified("; ".join(unverified))


def test_sample_count_property():
    H.check_night_sample_count_property()


def checks(ctx):
    return [
        ("contract.frozen_signatures", test_frozen_signatures),
        ("contract.sample_count_property", test_sample_count_property),
    ]
