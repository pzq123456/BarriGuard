"""MP4 source 验收: 与生产 server.source.Reader 接口等价 + 逐帧可读。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402


def test_reader_surface():
    H.check_reader_surface()


def test_mp4_reader():
    H.check_mp4_reader(None)


def checks(ctx):
    return [
        ("source.reader_surface", test_reader_surface),
        ("source.mp4_reader", lambda: H.check_mp4_reader(ctx.video)),
    ]
