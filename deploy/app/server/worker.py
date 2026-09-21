"""单相机工作循环 + Runtime 编排。

CameraWorker：Reader 拉流 -> 按 Scheduler 门控的算法 step -> 通用渲染/证据/事件。
Runtime     ：唯一持有 Scheduler 的编排者，用后台线程 poll() 并把
              NIGHT_START / NIGHT_FREEZE / NIGHT_FINALIZE / HOURLY_SNAPSHOT
              分发给各相机的 DayRunner / NightAdapter / Reporter。

边界（Wave 1）：
  * 本模块不实现任何算法：DAY 走 `server.agents.day.DayRunner`，
    NIGHT 走 `night_lamp.adapter.NightAdapter`，上报走 `server.reporting.Reporter`。
  * 依赖模块按冻结签名 import（懒加载，便于在依赖就绪前单测）；
    测试可通过 Runtime/CameraWorker 的 factory 注入替身。
"""
import dataclasses
import threading
import time
from collections import deque
from datetime import datetime

import cv2 as cv
from loguru import logger

from . import registry, render
from .schedule import (
    Clock, RealClock, RuntimeEvent, RuntimeEventType, Scheduler,
)
from .source import Reader

LOG_EVERY = 100
EVENTS_KEPT = 200
PUMP_INTERVAL_S = 0.5
STOP_JOIN_TIMEOUT_S = 5.0
STREAM_LOST_AFTER_S = 10.0
STREAM_HEARTBEAT_S = 600.0


def _default_day_runner(camera, spec, algo):
    from .agents.day import DayRunner
    return DayRunner(camera, spec, algo)


def _default_night_adapter(camera_id, spec, clock):
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    return NightAdapter(NightSession(camera_id, spec), spec, clock)


def _default_reporter(cfg):
    from .reporting import Reporter
    return Reporter(cfg)


def _default_store(cfg):
    """output_dir 为空时不落盘（测试/旧部署）；否则写 host-mounted 目录。"""
    out = str(getattr(cfg, "output_dir", "") or "").strip()
    if not out:
        return None
    from .output import ReportStore
    return ReportStore(out)


def _default_night_store(cfg):
    """夜间累计快照目录；output_dir 为空则不启用续跑。"""
    out = str(getattr(cfg, "output_dir", "") or "").strip()
    if not out:
        return None
    from night_lamp.persist import NightStateStore
    return NightStateStore(out)


def _default_night_archive(cfg):
    """夜间热力图核心数据（L1）归档目录；output_dir 为空则不启用。"""
    out = str(getattr(cfg, "output_dir", "") or "").strip()
    if not out:
        return None
    from night_lamp.persist import NightArchive
    return NightArchive(out)


# 业务策略字段 -> 算法 track 键（仅当 spec 暴露该字段时注入）。
_WATER_POLICY = (("confidence", "rer_threshold"),
                 ("alarm_hold_s", "alarm_hold_s"),
                 ("reconfirm_s", "reconfirm_s"))


def _apply_water_params(calib, spec):
    """把 WaterGapSpec 的 detect/track/策略合并进标定 dict（算法只读 calib）。

    水马数据文件只剩 rows/road_rois；detect、track 与 rer_threshold/
    alarm_hold_s/reconfirm_s 全由本函数从 spec 注入，保证单一来源。
    spec 无这些字段时（如夜灯）原样返回。
    """
    if not isinstance(calib, dict):
        return calib
    out = dict(calib)

    detect = getattr(spec, "detect", None)
    if dataclasses.is_dataclass(detect):
        out["detect"] = dataclasses.asdict(detect)

    track = dict(out.get("track") or {})
    track_spec = getattr(spec, "track", None)
    if dataclasses.is_dataclass(track_spec):
        track.update(dataclasses.asdict(track_spec))
    for attr, key in _WATER_POLICY:
        val = getattr(spec, attr, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            track[key] = val
    if track:
        out["track"] = track
    return out


def _load_calibration(name, spec):
    """spec.calibration 是（绝对）路径；也可能是已加载内容（dict），两者都兼容。"""
    cal = getattr(spec, "calibration", "")
    if not isinstance(cal, str):
        return _apply_water_params(cal, spec)
    from .config import ROOT
    from .config_validate import resolve_resource
    resolved = resolve_resource(cal, ROOT, f"algorithm.{name}.calibration")
    return _apply_water_params(registry.load_calib(name, resolved), spec)


class CameraWorker:
    def __init__(self, camera, scheduler: Scheduler, reporter,
                 jpeg_quality: int = 80, evidence=None,
                 clock: Clock | None = None,
                 day_factory=None, night_factory=None, store=None,
                 night_store=None, night_archive=None):
        self._camera = camera
        self.id = camera.id
        self.name = getattr(camera, "name", None) or camera.id
        self._scheduler = scheduler
        self._reporter = reporter
        self._store = store
        self._night_store = night_store
        self._night_archive = night_archive
        self._clock = clock or RealClock()
        self._quality = jpeg_quality
        self._evidence = evidence
        self._day_factory = day_factory or _default_day_runner
        self._night_factory = night_factory or _default_night_adapter

        self._reader = Reader(camera.rtsp_url)
        self._lock = threading.Lock()
        self._vis = None
        self._debug = {}
        self._status = "OK"
        self._events = deque(maxlen=EVENTS_KEPT)
        self._n = 0
        self._built = False
        self._day_runners: dict = {}
        self._night: dict = {}
        self._latest_frame = None
        self._stop = threading.Event()
        self._stream_lost = False
        self._stream_last_emit = None
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"cam-{self.id}")

    # ------------------------------------------------------------ 只读出口
    @property
    def camera(self):
        return self._camera

    @property
    def algo_names(self):
        return sorted((getattr(self._camera, "algorithms", None) or {}).keys())

    def start(self):
        self._stop.clear()
        self._reader.start()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._reader.stop()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=STOP_JOIN_TIMEOUT_S)

    def latest(self):
        with self._lock:
            return self._vis, dict(self._debug)

    def status(self):
        with self._lock:
            return self._status

    def debug_layer(self, layer):
        with self._lock:
            m = self._debug.get(layer)
            return None if m is None else m.copy()

    def events(self):
        with self._lock:
            return list(self._events)

    def publish(self, report) -> None:
        """落盘（若配置 output_dir）+ 异步 callback；任何失败都不冒泡。"""
        if report is None:
            return
        if self._store is not None:
            try:
                self._store.save(report)
            except Exception:
                logger.exception("[{}] report 落盘失败", self.id)
        try:
            self._reporter.submit(report)
        except Exception:
            logger.exception("[{}] report submit 失败", self.id)

    @staticmethod
    def _take_night_reports(adapter) -> list:
        take = getattr(adapter, "take_reports", None)
        if take is None:
            return []
        return take() or []

    # ------------------------------------------------------------ Runtime 回调
    def start_night(self):
        """NIGHT_START：为每个 schedule=night 的绑定建独立 NightAdapter。"""
        for name, binding in (getattr(self._camera, "algorithms", None) or {}).items():
            if getattr(binding, "schedule", None) != "night":
                continue
            with self._lock:
                if name in self._night:
                    continue
                try:
                    ad = self._night_factory(
                        self.id, binding.spec, self._clock)
                except Exception:
                    logger.exception("[{}] 创建 NightAdapter 失败: {}", self.id, name)
                    continue
                self._attach_night_state(ad, name)
                self._night[name] = ad
        logger.info("[{}] 夜间采样开始: {}", self.id, sorted(self._night))

    def _attach_night_state(self, adapter, name):
        """挂上夜间快照 store 与 L1 归档，并尝试恢复本夜累计（跨重启续跑）。"""
        attach = getattr(adapter, "attach_persistence", None)
        if callable(attach) and self._night_store is not None:
            try:
                attach(self._night_store)
            except Exception:
                logger.exception("[{}] 夜间持久化挂载失败: {}", self.id, name)
        attach_arch = getattr(adapter, "attach_archive", None)
        if callable(attach_arch) and self._night_archive is not None:
            try:
                attach_arch(self._night_archive)
            except Exception:
                logger.exception("[{}] 夜间归档挂载失败: {}", self.id, name)
        if self._night_store is None:
            return
        try:
            adapter.restore()
        except Exception:
            logger.exception("[{}] 夜间状态恢复失败: {}", self.id, name)

    def flush_night(self):
        """停机前把在跑的夜间累计落盘；best-effort，不冒泡。"""
        with self._lock:
            adapters = list(self._night.values())
        for ad in adapters:
            save = getattr(ad, "save_state", None)
            if not callable(save):
                continue
            try:
                save()
            except Exception:
                logger.exception("[{}] night flush 失败", self.id)

    def _check_stream_health(self, now_mono: float):
        """检测断流/恢复并推状态事件；Reader 无 health() 时静默跳过。"""
        health = getattr(self._reader, "health", None)
        if not callable(health):
            return
        info = health()
        if info.get("connected"):
            if self._stream_lost:
                self._stream_lost = False
                self._stream_last_emit = now_mono
                self._set_status("OK")
                self._emit_status("stream_restored", info)
            return
        ref = info.get("last_frame_mono") or info.get("started_mono")
        if ref is None or now_mono - ref < STREAM_LOST_AFTER_S:
            return
        if not self._stream_lost:
            self._stream_lost = True
            self._stream_last_emit = now_mono
            self._set_status("STREAM_LOST")
            self._emit_status("stream_lost", info)
            return
        if now_mono - (self._stream_last_emit or now_mono) >= STREAM_HEARTBEAT_S:
            self._stream_last_emit = now_mono
            self._emit_status("stream_lost", info)

    def _set_status(self, status: str):
        with self._lock:
            self._status = status

    def _emit_status(self, status: str, health: dict):
        detail = {
            "reconnects": health.get("reconnects"),
            "seconds_since_frame": health.get("seconds_since_frame"),
        }
        submit = getattr(self._reporter, "submit_status", None)
        if callable(submit):
            try:
                submit(self.id, status, detail=detail)
            except Exception:
                logger.exception("[{}] 状态上报失败: {}", self.id, status)
        with self._lock:
            self._events.append({"ts": datetime.now().astimezone().isoformat(),
                                 "algo": "stream", "kind": "status",
                                 "payload": {"status": status},
                                 "evidence_path": None})
        logger.warning("[{}] 流状态: {}", self.id, status)

    def freeze_night(self):
        """NIGHT_FREEZE：停止累积（各 session 冻结）。"""
        with self._lock:
            adapters = list(self._night.values())
        for ad in adapters:
            try:
                ad.freeze()
            except Exception:
                logger.exception("[{}] night freeze 失败", self.id)
        logger.info("[{}] 夜间采样冻结", self.id)

    def finalize_night(self, ts_wall: datetime):
        """NIGHT_FINALIZE：latest_frame -> finalize -> submit -> release（按夜隔离）。"""
        with self._lock:
            adapters = list(self._night.items())
            base = None if self._latest_frame is None else self._latest_frame.copy()
        for name, ad in adapters:
            try:
                report = ad.finalize(base, ts_wall)
                if report is not None:
                    self.publish(report)
            except Exception:
                logger.exception("[{}] night finalize 失败: {}", self.id, name)
            finally:
                try:
                    ad.release()
                except Exception:
                    logger.exception("[{}] night release 失败: {}", self.id, name)
                with self._lock:
                    self._night.pop(name, None)
        logger.info("[{}] 夜间 finalize 完成并 release", self.id)

    def snapshot_reports(self, ts_wall: datetime) -> list:
        """HOURLY_SNAPSHOT：对含 report_interval_s 且当前活跃的白昼算法出图。"""
        out = []
        for name, runner in list(self._day_runners.items()):
            binding = (getattr(self._camera, "algorithms", None) or {}).get(name)
            spec = getattr(binding, "spec", None)
            if not getattr(spec, "report_interval_s", None):
                continue
            if not self._scheduler.is_active(self.id, name):
                continue
            try:
                rep = runner.snapshot(ts_wall)
            except Exception:
                logger.exception("[{}] snapshot 失败: {}", self.id, name)
                continue
            if rep is not None:
                out.append(rep)
        return out

    # ------------------------------------------------------------ 工作循环
    def _build(self, shape):
        for name, binding in (getattr(self._camera, "algorithms", None) or {}).items():
            if getattr(binding, "schedule", None) != "day":
                continue
            try:
                algo = registry.create(name, shape,
                                       _load_calibration(name, binding.spec), self.id)
                self._day_runners[name] = self._day_factory(
                    self._camera, binding.spec, algo)
            except Exception:
                logger.exception("[{}] 构建 DayRunner 失败: {}", self.id, name)
        self._built = True
        logger.info("[{}] 已加载白昼算法: {}", self.id, sorted(self._day_runners))

    def _loop(self):
        t0, n0 = time.time(), 0
        while not self._stop.is_set():
            frame = self._reader.read()
            self._check_stream_health(self._clock.monotonic())
            if frame is None:
                time.sleep(0.05)
                continue
            if not self._built:
                self._build(frame.shape)
            now_wall = self._clock.wall()
            now_mono = self._clock.monotonic()
            self._latest_frame = frame

            annots, debug, events = [], {}, []
            status = "OK"
            for name, runner in list(self._day_runners.items()):
                if not self._scheduler.is_active(self.id, name):
                    continue
                try:
                    res = runner.on_frame(frame, now_mono)
                except Exception:
                    logger.exception("[{}] day step 失败: {}", self.id, name)
                    continue
                annots += res.annots
                debug.update(res.debug)
                events += res.events
                for rep in getattr(res, "reports", None) or ():
                    try:
                        self.publish(rep)
                    except Exception:
                        logger.exception("[{}] publish report 失败: {}",
                                         self.id, name)
                if isinstance(res.debug.get("frame_status"), str):
                    status = res.debug["frame_status"]

            with self._lock:
                adapters = list(self._night.items())
            for name, ad in adapters:
                if not self._scheduler.is_active(self.id, name):
                    continue
                try:
                    ad.on_frame(frame, now_wall, now_mono)
                except Exception:
                    logger.exception("[{}] night on_frame 失败: {}", self.id, name)
                    continue
                for rep in self._take_night_reports(ad):
                    self.publish(rep)

            mask = debug.get("roi_mask")
            vis = render.draw_annots(
                render.overlay(frame, mask) if mask is not None else frame, annots)
            vis = render.draw_status(vis, status)
            alarming = any(e.kind == "alarm" for e in events)
            states = "+".join(sorted({a.level for a in annots}))
            path = (self._evidence.maybe_save(self.id, vis, alarming, states, now_mono)
                    if self._evidence is not None else None)
            if path:
                for e in events:
                    if e.kind == "alarm" and e.evidence_path is None:
                        e.evidence_path = path
            ok, buf = cv.imencode(".jpg", vis, [cv.IMWRITE_JPEG_QUALITY, self._quality])
            jpeg_bytes = buf.tobytes() if ok else None
            for e in events:
                if e.kind != "alarm":
                    continue
                try:
                    self._reporter.submit_event(e, image_jpeg=jpeg_bytes)
                except Exception:
                    logger.exception("[{}] submit_event 失败", self.id)
            with self._lock:
                self._vis = buf.tobytes() if ok else None
                self._debug = debug
                self._status = status
                self._events.extend({"ts": e.ts, "algo": e.algo, "kind": e.kind,
                                     "payload": e.payload,
                                     "evidence_path": e.evidence_path} for e in events)
                self._n += 1
            if self._n - n0 >= LOG_EVERY:
                elapsed = max(time.time() - t0, 1e-6)
                logger.info("[{}] 处理 {} 帧, {:.1f} fps", self.id, self._n,
                            (self._n - n0) / elapsed)
                t0, n0 = time.time(), self._n


class Runtime:
    """唯一 Scheduler 持有者：poll -> 分发事件到 workers / Reporter。"""

    def __init__(self, cfg, clock: Clock | None = None, reporter=None,
                 evidence=None,
                 day_factory=None, night_factory=None):
        self.cfg = cfg
        self.clock = clock or RealClock()
        self.scheduler = Scheduler(cfg, self.clock)
        self.reporter = reporter if reporter is not None else _default_reporter(cfg)
        quality = 80
        if isinstance(getattr(cfg, "server", None), dict):
            quality = int(cfg.server.get("jpeg_quality", 80))
        self.store = _default_store(cfg)
        self.night_store = _default_night_store(cfg)
        self.night_archive = _default_night_archive(cfg)
        self.workers = {}
        for cam in cfg.cameras or []:
            if not getattr(cam, "enabled", True):
                continue
            self.workers[cam.id] = CameraWorker(
                cam, self.scheduler, self.reporter, quality, evidence=evidence,
                clock=self.clock, day_factory=day_factory,
                night_factory=night_factory, store=self.store,
                night_store=self.night_store,
                night_archive=self.night_archive)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        for w in self.workers.values():
            w.start()
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name="runtime-pump")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        for w in self.workers.values():
            try:
                w.flush_night()
            except Exception:
                logger.exception("night flush 失败")
        for w in self.workers.values():
            w.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=PUMP_INTERVAL_S * 4)
        try:
            self.reporter.close()
        except Exception:
            logger.exception("reporter.close 失败")

    close = stop

    def _pump(self):
        while not self._stop.wait(PUMP_INTERVAL_S):
            try:
                events = self.scheduler.poll()
            except Exception:
                logger.exception("scheduler.poll 失败")
                continue
            for ev in events:
                try:
                    self.dispatch(ev)
                except Exception:
                    logger.exception("runtime 事件处理失败: {}", ev.type)

    def dispatch(self, ev: RuntimeEvent):
        if ev.type == RuntimeEventType.NIGHT_START:
            for w in self.workers.values():
                w.start_night()
        elif ev.type == RuntimeEventType.NIGHT_FREEZE:
            for w in self.workers.values():
                w.freeze_night()
        elif ev.type == RuntimeEventType.NIGHT_FINALIZE:
            for w in self.workers.values():
                w.finalize_night(ev.at)
        elif ev.type == RuntimeEventType.HOURLY_SNAPSHOT:
            w = self.workers.get(ev.camera)
            if w is None:
                return
            for rep in w.snapshot_reports(ev.at):
                w.publish(rep)
