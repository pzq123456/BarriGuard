"""可靠性验收(上线阻断组): 夜间续跑、告警落盘、优雅停机。

不依赖 RTSP/MP4，用合成帧与临时目录，确定性、可重复。
"""
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "app"
_TESTS = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_REPO = Path(__file__).resolve().parents[3]
_DAY_FRAME = _REPO / "data" / "1749" / "1749_20260917_155840_day.jpg"

import replay_harness as H  # noqa: E402

TZ = timezone.utc


def _night_spec(cadence_minutes=60, burst_seconds=120):
    from server.contracts import (AlignmentSpec, BaselineSpec, MemorySpec,
                                  NightLampSpec, OvernightSpec, PeriodicitySpec,
                                  SamplingSpec)

    return NightLampSpec(
        sampling=SamplingSpec(interval_ms=160),
        memory=MemorySpec(max_candidates=200, series_cap=None),
        baseline=BaselineSpec(frames=5, warmup_s=0),
        periodicity=PeriodicitySpec(onset_min=50),
        alignment=AlignmentSpec(),
        overnight=OvernightSpec(enabled=True, cadence_minutes=cadence_minutes,
                                burst_seconds=burst_seconds),
    )


def _frame(i):
    img = np.full((48, 64, 3), 40, np.uint8)
    if i % 2 == 0:
        img[10:20, 10:20] = 220
    return img


def _feed(session, start, stop):
    for i in range(start, stop):
        session.accumulate(_frame(i), i * 0.16)


def night_resume(ctx=H.Context()):
    """进程重启后，空间累计与样本计数续跑，07:00 夜图不再为空。"""
    from night_lamp.persist import NightStateStore
    from night_lamp.session import NightSession

    with tempfile.TemporaryDirectory() as tmp:
        spec = _night_spec()
        s1 = NightSession("cam", spec)
        _feed(s1, 0, 20)
        before = s1.sample_count
        state = s1.dump_state()
        if state is None:
            raise AssertionError("dump_state returned None after warmup")
        store = NightStateStore(tmp)
        if store.save("cam", "2026-09-18", state) is None:
            raise AssertionError("state save failed")
        s1.release()

        s2 = NightSession("cam", spec)
        loaded = s2.load_state(store.load("cam", "2026-09-18"))
        if not loaded:
            raise AssertionError("state load failed")
        if s2.sample_count != before:
            raise AssertionError(
                f"restored sample_count {s2.sample_count} != {before}")
        _feed(s2, 20, 40)
        rep = s2.finalize(None, datetime(2026, 9, 19, 7, 0, tzinfo=TZ))
        if rep is None or not rep.image_jpeg:
            raise AssertionError("finalize after restore produced no heatmap")
        if rep.metadata["sampling"]["actual_samples"] != 40:
            raise AssertionError(
                "cumulative samples lost: "
                f"{rep.metadata['sampling']['actual_samples']}")
        s2.release()
    return "restored and finalized"


def adapter_resume(ctx=H.Context()):
    """NightAdapter 通过 store 保存/恢复，不改变冻结签名。"""
    from night_lamp.adapter import NightAdapter
    from night_lamp.persist import NightStateStore
    from night_lamp.session import NightSession
    from server.schedule import FakeClock

    with tempfile.TemporaryDirectory() as tmp:
        spec = _night_spec()
        clock = FakeClock(datetime(2026, 9, 18, 20, 0, tzinfo=TZ))
        store = NightStateStore(tmp)

        s1 = NightSession("cam", spec)
        ad1 = NightAdapter(s1, spec, clock)
        ad1.attach_persistence(store)
        if ad1.restore():
            raise AssertionError("restore succeeded on empty store")
        _feed(s1, 0, 12)
        if not ad1.save_state(clock.wall()):
            raise AssertionError("adapter save_state failed")
        s1.release()

        s2 = NightSession("cam", spec)
        ad2 = NightAdapter(s2, spec, clock)
        ad2.attach_persistence(store)
        if not ad2.restore():
            raise AssertionError("adapter restore failed")
        if s2.sample_count != 12:
            raise AssertionError(f"restored samples {s2.sample_count} != 12")
        s2.release()
    return "adapter save/restore ok"


def alarm_persisted(ctx=H.Context()):
    """告警帧在服务端落盘，消费端不可达也留下 jpg+json。"""
    from server.algo import Event
    from server.contracts import RuntimeConfig
    from server.reporting import Reporter

    with tempfile.TemporaryDirectory() as tmp:
        cfg = RuntimeConfig(output_dir=tmp)
        reporter = Reporter(cfg)
        try:
            event = Event(camera_id="cam", algo="water_gap", ts=0.0,
                          kind="alarm",
                          payload={"row_id": "r0", "rer": 0.62,
                                   "box": [1, 2, 3, 4]})
            reporter.submit_event(event, image_jpeg=b"\xff\xd8frame")
            time.sleep(0.2)
        finally:
            reporter.close()

        metas = sorted((Path(tmp) / "alerts" / "cam").glob("*.json"))
        if not metas:
            raise AssertionError("alarm metadata not persisted")
        if not metas[0].with_suffix(".jpg").is_file():
            raise AssertionError("alarm frame jpg not persisted")
        text = metas[0].read_text(encoding="utf-8")
        for token in ("water_gap", "r0", "0.62"):
            if token not in text:
                raise AssertionError(f"payload missing {token!r}: {text}")
    return "alarm frame on disk"


def frozen_restart_rebuilds(ctx=H.Context()):
    """03:00–07:00 之间重启：补建并冻结会话，07:00 仍能 finalize。"""
    from server.schedule import FakeClock, Scheduler

    cfg = H.build_night_only_cfg()
    clock = FakeClock(datetime(2026, 9, 19, 5, 0,
                               tzinfo=H.resolve_tz(H.DEFAULT_TZ)))
    scheduler = Scheduler(cfg, clock)
    first = [H._kind(e) for e in scheduler.poll()]
    if "night_start" not in first or "night_freeze" not in first:
        raise AssertionError(f"FROZEN startup events wrong: {first}")
    clock.advance(2 * 3600)
    later = [H._kind(e) for e in scheduler.poll()]
    if "night_finalize" not in later:
        raise AssertionError(f"no finalize after freeze restart: {later}")
    return "frozen restart rebuilt"


class _HealthReader:
    def __init__(self, url):
        self.connected = False
        self.started_mono = 0.0
        self.last_frame_mono = None
        self.reconnects = 0

    def start(self):
        return self

    def stop(self):
        pass

    def join(self, timeout=None):
        pass

    def read(self):
        return None

    def health(self):
        return {
            "connected": self.connected,
            "started_mono": self.started_mono,
            "last_frame_mono": self.last_frame_mono,
            "seconds_since_frame": (None if self.last_frame_mono is None else 0.0),
            "reconnects": self.reconnects,
        }


def stream_health_alert(ctx=H.Context()):
    """持续无帧超阈值 -> 内部状态 STREAM_LOST；恢复 -> OK（不再对外推 camera_status）。"""
    import server.worker as W
    from server.contracts import CameraSpec
    from server.schedule import FakeClock, Scheduler

    original = W.Reader
    W.Reader = _HealthReader
    try:
        cfg = H.build_night_only_cfg()
        clock = FakeClock(datetime(2026, 9, 18, 19, 59, tzinfo=TZ))
        cam = CameraSpec(id="cam", name="cam", rtsp_url="fake://x",
                         algorithms={})
        reporter = H.NullReporter()
        worker = W.CameraWorker(cam, Scheduler(cfg, clock), reporter)

        worker._check_stream_health(5.0)
        if worker.status() != "OK":
            raise AssertionError("false stream_lost before threshold")
        worker._check_stream_health(11.0)
        if worker.status() != "STREAM_LOST":
            raise AssertionError(f"status={worker.status()}")

        worker._reader.connected = True
        worker._reader.last_frame_mono = 12.0
        worker._check_stream_health(12.0)
        if worker.status() != "OK":
            raise AssertionError(f"status={worker.status()}")
    finally:
        W.Reader = original
    return "stream status tracked internally"


def payload_contract(ctx=H.Context()):
    """Alarm 带 kind/target/bbox/objects/metadata；Report 带 status/report_type/metadata。"""
    import json as _json
    from server.algo import Event, Report
    from server.contracts import RuntimeConfig
    from server.reporting import Reporter

    reporter = Reporter(RuntimeConfig(output_dir=tempfile.mkdtemp()))
    try:
        event = Event(camera_id="cam", algo="water_gap", ts=0.0, kind="alarm",
                      payload={"row_id": "row_2", "severity": 51, "rer": 0.62,
                               "box": [10, 20, 30, 40]})
        body = _json.loads(reporter._body("event", event, b"\xff\xd8x"))
        if body.get("kind") != "alarm" or body.get("target") != "row_2":
            raise AssertionError(f"event identity missing: {body}")
        if body.get("bbox") != [10, 20, 30, 40]:
            raise AssertionError(f"event bbox missing: {body}")
        if not body.get("objects") or body["objects"][0]["confidence"] != 0.62:
            raise AssertionError(f"event objects missing: {body}")
        if body.get("metadata", {}).get("severity") != 51:
            raise AssertionError(f"event metadata missing: {body}")

        rep = Report(camera="cam", algorithm="night_lamp",
                     report_type="night_heatmap",
                     created_at="2026-09-19T07:00:00+08:00",
                     image_jpeg=b"\xff\xd8x", status="degraded",
                     metadata={"candidate_overflow": True})
        rbody = _json.loads(reporter._body("report", rep, None))
        if rbody.get("status") != "degraded":
            raise AssertionError(f"report status missing: {rbody}")
        if rbody.get("report_type") != "night_heatmap":
            raise AssertionError(f"report_type missing: {rbody}")
        if not rbody.get("metadata", {}).get("candidate_overflow"):
            raise AssertionError(f"report metadata missing: {rbody}")
    finally:
        reporter.close()
    return "payloads complete"


def output_night_images(ctx=H.Context()):
    """真实夜灯管线跑 MP4，经 ReportStore 把热力图 jpg+json 落到磁盘。"""
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server.output import ReportStore
    from server.schedule import FakeClock

    with tempfile.TemporaryDirectory() as tmp:
        spec = _night_spec(cadence_minutes=1, burst_seconds=20)
        reader = H.Mp4Reader(H.ensure_video(ctx.video), loop=True).start()
        clock = FakeClock(datetime(2026, 9, 18, 20, 0,
                                   tzinfo=H.resolve_tz(H.DEFAULT_TZ)))
        session = NightSession("cam", spec)
        adapter = NightAdapter(session, spec, clock)
        step = spec.sampling.interval_ms / 1000.0 * 1.05
        t = 0.0
        try:
            while t < 25.0:
                frame = reader.read()
                if frame is None:
                    break
                clock.advance(step)
                t += step
                adapter.on_frame(frame, clock.wall(), clock.monotonic())
            reports = list(adapter.take_reports())
            adapter.freeze()
            final = adapter.finalize(None, clock.wall())
        finally:
            adapter.release()
            reader.stop()

        store = ReportStore(tmp)
        for rep in reports + [final]:
            if rep is not None:
                store.save(rep)

        jpgs = sorted(Path(tmp).rglob("*.jpg"))
        if not jpgs:
            raise AssertionError("no report image on disk")
        if jpgs[0].read_bytes()[:2] != b"\xff\xd8":
            raise AssertionError("written file is not a JPEG")
        if not list(Path(tmp).rglob("*.json")):
            raise AssertionError("no report metadata json on disk")
    return f"{len(jpgs)} images on disk"


def water_gap_image_on_disk(ctx=H.Context()):
    """真实 water_gap 算法跑真实白天帧，产出叠加图并落盘。"""
    import cv2 as cv
    from server import registry
    from server.agents.day import DayRunner
    from server.contracts import CameraSpec, WaterGapSpec
    from server.output import ReportStore
    from server.worker import _load_calibration

    if not _DAY_FRAME.is_file():
        raise H.Unverified(f"day frame missing: {_DAY_FRAME}")

    with tempfile.TemporaryDirectory() as tmp:
        frame = cv.imread(str(_DAY_FRAME))
        if frame is None:
            raise H.Unverified("cv2 failed to read day frame")
        spec = WaterGapSpec(calibration="water_barrier/configs/1749.yaml")
        cam = CameraSpec(id="1749", name="1749", rtsp_url="file://day.jpg")
        algo = registry.create("water_gap", frame.shape,
                               _load_calibration("water_gap", spec), "1749")
        runner = DayRunner(cam, spec, algo)
        for i in range(14):
            runner.on_frame(frame, float(i))
        rep = runner.snapshot(datetime(2026, 9, 17, 8, 0, tzinfo=TZ))
        if rep is None or rep.image_jpeg[:2] != b"\xff\xd8":
            raise AssertionError("water_gap snapshot produced no JPEG")
        store = ReportStore(tmp)
        path = store.save(rep)
        if not Path(path).is_file():
            raise AssertionError("water_gap image not written")
    return "water_gap overlay on disk"


def archive_heatmap_core(ctx=H.Context()):
    """L1 归档：热力图核心数组（on/peak/base/bg/duty/swing）按夜落盘且不清理。"""
    from night_lamp.adapter import NightAdapter
    from night_lamp.persist import NightArchive, NightStateStore
    from night_lamp.session import NightSession
    from server.schedule import FakeClock

    with tempfile.TemporaryDirectory() as tmp:
        spec = _night_spec()
        clock = FakeClock(datetime(2026, 9, 18, 20, 0,
                                   tzinfo=H.resolve_tz(H.DEFAULT_TZ)))
        session = NightSession("cam", spec)
        adapter = NightAdapter(session, spec, clock)
        adapter.attach_persistence(NightStateStore(tmp))
        adapter.attach_archive(NightArchive(tmp))
        _feed(session, 0, 12)
        adapter.save_state(clock.wall())
        adapter.finalize(None, clock.wall())

        arch = Path(tmp) / "night_archive"
        npzs = sorted(arch.rglob("state.npz"))
        if not npzs:
            raise AssertionError("no archived heatmap core")
        with np.load(npzs[0]) as z:
            for key in ("on", "peak", "base", "bg", "n_seen", "duty", "swing"):
                if key not in z.files:
                    raise AssertionError(f"archive missing {key}: {z.files}")
        if not list(arch.rglob("manifest.json")):
            raise AssertionError("no manifest.json in archive")
        session.release()
    return "heatmap core archived"


class _FakeReader:
    def __init__(self, url):
        self._stop = threading.Event()

    def start(self):
        return self

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        pass

    def read(self):
        if self._stop.is_set():
            return None
        return np.zeros((8, 8, 3), np.uint8)


def graceful_stop(ctx=H.Context()):
    """CameraWorker.stop 能终止工作线程，不再 while True 泄漏。"""
    import server.worker as W
    from server.contracts import CameraSpec
    from server.schedule import FakeClock, Scheduler

    original = W.Reader
    W.Reader = _FakeReader
    worker = None
    try:
        cfg = H.build_night_only_cfg()
        clock = FakeClock(datetime(2026, 9, 18, 19, 59, tzinfo=TZ))
        cam = CameraSpec(id="cam", name="cam", rtsp_url="fake://x",
                         algorithms={})
        worker = W.CameraWorker(cam, Scheduler(cfg, clock), H.NullReporter())
        worker.start()
        time.sleep(0.15)
        if not worker._thread.is_alive():
            raise AssertionError("worker thread did not start")
        worker.stop()
        if worker._thread.is_alive():
            raise AssertionError("worker thread still alive after stop")
    finally:
        if worker is not None and worker._thread.is_alive():
            worker.stop()
        W.Reader = original
    return "thread joined on stop"


def test_night_resume():
    night_resume()


def test_adapter_resume():
    adapter_resume()


def test_alarm_persisted():
    alarm_persisted()


def test_frozen_restart_rebuilds():
    frozen_restart_rebuilds()


def test_stream_health_alert():
    stream_health_alert()


def test_payload_contract():
    payload_contract()


def test_output_night_images():
    output_night_images()


def test_water_gap_image_on_disk():
    water_gap_image_on_disk()


def test_archive_heatmap_core():
    archive_heatmap_core()


def test_graceful_stop():
    graceful_stop()


def day_ref_base(ctx=H.Context()):
    """夜间 finalize 必须优先用缓存的白天基准帧，而不是 07:00 的实时暗帧。"""
    import server.worker as W
    from server.contracts import CameraSpec
    from server.schedule import FakeClock, Scheduler

    original = W.Reader
    W.Reader = _FakeReader
    try:
        cfg = H.build_night_only_cfg("UTC")
        clock = FakeClock(datetime(2026, 9, 18, 10, 0, tzinfo=TZ))
        sched = Scheduler(cfg, clock)
        sched.poll()  # 让状态进入 DAY
        cam = CameraSpec(id="cam", name="cam", rtsp_url="fake://x", algorithms={})
        worker = W.CameraWorker(cam, sched, H.NullReporter())

        dark = np.full((8, 8, 3), 10, np.uint8)
        bright = np.full((8, 8, 3), 200, np.uint8)

        worker._maybe_capture_day_ref(dark, 0.0)
        if worker._day_ref is not None:
            raise AssertionError("暗帧不应被选作白天基准")
        worker._latest_frame = dark
        if worker._select_day_base()[1] != "live":
            raise AssertionError("无缓存时应回落到实时帧")

        worker._maybe_capture_day_ref(bright, 1.0)
        frame, src = worker._select_day_base()
        if src != "day_ref" or frame is None or float(frame.mean()) < 100:
            raise AssertionError(f"应优先用白天基准帧, got src={src}")
    finally:
        W.Reader = original
    return "finalize base prefers cached day frame"


def checks(ctx):
    return [
        ("reliability.night_resume", lambda: night_resume(ctx)),
        ("reliability.day_ref_base", lambda: day_ref_base(ctx)),
        ("reliability.adapter_resume", lambda: adapter_resume(ctx)),
        ("reliability.alarm_persisted", lambda: alarm_persisted(ctx)),
        ("reliability.frozen_restart_rebuilds",
         lambda: frozen_restart_rebuilds(ctx)),
        ("reliability.stream_health_alert",
         lambda: stream_health_alert(ctx)),
        ("reliability.payload_contract", lambda: payload_contract(ctx)),
        ("reliability.output_night_images", lambda: output_night_images(ctx)),
        ("reliability.water_gap_image_on_disk",
         lambda: water_gap_image_on_disk(ctx)),
        ("reliability.archive_heatmap_core",
         lambda: archive_heatmap_core(ctx)),
        ("reliability.graceful_stop", lambda: graceful_stop(ctx)),
    ]
