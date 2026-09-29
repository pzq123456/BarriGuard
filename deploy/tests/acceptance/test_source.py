"""MP4 source 验收: 与生产 server.source.Reader 接口等价 + 逐帧可读。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "app"
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402


def test_reader_surface():
    H.check_reader_surface()


def test_mp4_reader():
    H.check_mp4_reader(None)


def grey_guard(ctx=H.Context()):
    """灰帧自愈的判据：近常量帧判灰，有纹理的帧不判灰。"""
    import numpy as np

    from server.source import Reader

    r = Reader("file://x")
    flat = np.full((1080, 1920, 3), 128, np.uint8)
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (1080, 1920, 3), np.uint8)
    if not r._is_grey(flat):
        raise AssertionError("常量帧应判为灰帧")
    if r._is_grey(textured):
        raise AssertionError("有纹理帧不应判为灰帧")
    return "grey guard flags constant frames"


def checks(ctx):
    return [
        ("source.reader_surface", test_reader_surface),
        ("source.mp4_reader", lambda: H.check_mp4_reader(ctx.video)),
        ("source.grey_guard", lambda: grey_guard(ctx)),
    ]
