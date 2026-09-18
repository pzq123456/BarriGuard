"""白昼 DayRunner 验收: MP4 逐帧透传 + 内存快照 Report (stub 算法，不载模型)。

这里验证的是 source->DayRunner->Report 的管线与冻结契约，不是水马算法精度；
生产算法(WaterGapAlgorithm)的验证归 Agent C。
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import replay_harness as H  # noqa: E402
from server.algo import AlgoResult, Report  # noqa: E402
from server.contracts import CameraSpec, WaterGapSpec  # noqa: E402


class StubAlgo:
    name = "water_gap"
    cadence = "per_frame"

    def step(self, frame, now):
        return AlgoResult(events=[], annots=[], debug={}, reports=[])

    def reset(self):
        pass


def _run(video):
    H.require_modules([("server.agents.day", "DayRunner")])
    from server.agents.day import DayRunner

    cam = CameraSpec(id=H.CAMERA_ID, name="replay-day",
                     rtsp_url="file://replay.mp4")
    runner = DayRunner(cam, WaterGapSpec(schedule="day"), StubAlgo())

    if runner.snapshot(datetime.now(timezone.utc)) is not None:
        raise AssertionError("snapshot before any frame must be None")

    reader = H.Mp4Reader(H.ensure_video(video), loop=True).start()
    try:
        frames = reader.burst_next(3)
        for i, frame in enumerate(frames):
            res = runner.on_frame(frame, 100.0 + i)
            if not isinstance(res, AlgoResult):
                raise AssertionError(f"on_frame returned {type(res).__name__}")
            if res.reports:
                raise AssertionError("stub algo must not be forced to emit Report")
    finally:
        reader.stop()

    report = runner.snapshot(datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc))
    if not isinstance(report, Report):
        raise AssertionError("snapshot must return Report after frames")
    if report.report_type != "gap_overlay":
        raise AssertionError(f"report_type={report.report_type}")
    if report.camera != H.CAMERA_ID or report.algorithm != "water_gap":
        raise AssertionError("report camera/algorithm mismatch")
    if not report.created_at.startswith("2026-09-18T12:00"):
        raise AssertionError(f"created_at={report.created_at}")
    if not report.image_jpeg or report.image_jpeg[:2] != b"\xff\xd8":
        raise AssertionError("snapshot missing in-memory JPEG")
    return f"jpeg={len(report.image_jpeg)}B"


def test_day_runner_plumbing():
    _run(None)


def checks(ctx):
    return [("day.runner_plumbing", lambda: _run(ctx.video))]
