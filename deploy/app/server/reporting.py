"""上报模块：把算法成品（Report）与告警（Event）异步投递到 callback。

新 Runtime 用 :class:`Reporter`（从 ``RuntimeConfig`` 读 ``callback.*``）：
- 后台 daemon 线程 + **有界**队列（``callback.queue_size``），绝不阻塞分析循环；
- HTTP 失败不阻塞、不抛出，允许丢弃并打告警日志；
- Report 图片按 ``callback.image_field`` 走 base64；
- callback 超时用 ``callback.timeout_s``；
- 告警按 ``(camera, algo, target)`` 节流，间隔取 ``WaterGapSpec.alarm_min_interval_s``。

本模块还保留旧的模块级 ``notify()`` 兼容 shim（由尚未迁移的
``server/worker.py`` 调用，走 ``WEBHOOK_URL`` 环境变量），新 Runtime 不应再用它。
"""
import base64
import json
import os
import queue
import threading
import time
import urllib.request
from datetime import datetime

from loguru import logger

from .algo import Event, Report  # noqa: F401  (Report/Event 契约复用)
from .contracts import RuntimeConfig, WaterGapSpec

_DEFAULT_ALARM_MIN_INTERVAL_S = 3600.0
_DEFAULT_IMAGE_FIELD = "image_base64"
_CLOSE_JOIN_TIMEOUT_S = 5.0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _target(e) -> str:
    """告警目标标识：夜灯用 lamp_id，水马用 row_id，兜底 target。"""
    payload = getattr(e, "payload", None) or {}
    for key in ("lamp_id", "row_id", "target"):
        val = payload.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


def _bbox(e):
    payload = getattr(e, "payload", None) or {}
    box = payload.get("box")
    if box:
        return list(box)
    x, y = payload.get("x"), payload.get("y")
    if x is not None and y is not None:
        return [x - 8, y - 8, x + 8, y + 8]
    return []


class Reporter:
    """异步、有界、best-effort 的 callback 上报器。"""

    def __init__(self, cfg: RuntimeConfig):
        cb = cfg.callback
        self._cb = cb
        self._url = (cb.url or "").strip()
        self._timeout = float(cb.timeout_s)
        self._image_field = (cb.image_field or _DEFAULT_IMAGE_FIELD).strip() \
            or _DEFAULT_IMAGE_FIELD
        self._enabled = bool(cb.enabled and self._url)

        queue_size = max(1, int(cb.queue_size))
        self._q = queue.Queue(maxsize=queue_size)

        self._lock = threading.Lock()
        self._last = {}
        self._closed = False
        self._dropped = 0
        self._last_drop_log = 0.0

        self._alarm_interval = {}
        self._camera_name = {}
        for cam in getattr(cfg, "cameras", None) or []:
            cid = getattr(cam, "id", None)
            if cid is None:
                continue
            self._camera_name[cid] = getattr(cam, "name", None) or cid
            for name, binding in (getattr(cam, "algorithms", None) or {}).items():
                spec = getattr(binding, "spec", None)
                interval = getattr(spec, "alarm_min_interval_s", None)
                if interval is not None:
                    self._alarm_interval[(cid, name)] = float(interval)

        self._thread = None
        if self._enabled:
            self._thread = threading.Thread(target=self._drain, daemon=True,
                                            name="reporter")
            self._thread.start()
        else:
            logger.info("[reporter] callback 未启用或 url 为空，整体 no-op")

    # ---------------------------------------------------------------- public
    def submit(self, report: Report) -> None:
        """入队一条图片 Report；永不阻塞、永不抛出。"""
        if not self._enabled or self._closed:
            return
        self._enqueue(("report", report))

    def submit_event(self, event) -> None:
        """入队一条告警 Event（带 (camera, algo, target) 节流）。"""
        if not self._enabled or self._closed:
            return
        if getattr(event, "kind", None) != "alarm":
            return
        key = (getattr(event, "camera_id", ""), getattr(event, "algo", ""),
               _target(event))
        interval = self._alarm_interval.get((key[0], key[1]),
                                            _DEFAULT_ALARM_MIN_INTERVAL_S)
        now = time.monotonic()
        with self._lock:
            if now - self._last.get(key, 0.0) < interval:
                return
            self._last[key] = now
        self._enqueue(("event", event))

    def close(self) -> None:
        """停止后台线程；幂等，最多等待 ``_CLOSE_JOIN_TIMEOUT_S``。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if not self._enabled or self._thread is None:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)

    # --------------------------------------------------------------- private
    def _enqueue(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            now = time.monotonic()
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
                should_log = now - self._last_drop_log >= 5.0
                if should_log:
                    self._last_drop_log = now
            if should_log:  # 限流日志，避免持续故障时刷爆
                logger.warning("[reporter] 队列满（size={}），丢弃 {}；累计丢弃 {}",
                               self._q.maxsize, item[0], dropped)
        except Exception as e:  # 防御：入队路径绝不冒泡
            logger.warning("[reporter] 入队异常: {}: {}", type(e).__name__, e)

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                self._send(item)
            except Exception as e:  # 单条失败不得杀线程
                logger.warning("[reporter] 发送异常: {}: {}", type(e).__name__, e)
            finally:
                self._q.task_done()

    def _send(self, item) -> None:
        kind, payload = item
        body = (self._body_report(payload) if kind == "report"
                else self._body_event(payload))
        if body is None:
            return
        req = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                logger.info("[reporter] {} -> {} {}", kind, self._url, r.status)
        except Exception as e:
            logger.warning("[reporter] 上报失败({}): {}: {}",
                           kind, type(e).__name__, e)

    def _body_report(self, report: Report) -> bytes | None:
        obj = {
            "type": "report",
            "camera_id": report.camera,
            "camera_name": self._camera_name.get(report.camera, report.camera),
            "algorithm": report.algorithm,
            "report_type": report.report_type,
            "created_at": report.created_at,
            "status": report.status,
            "metadata": report.metadata,
        }
        if report.image_jpeg:
            obj[self._image_field] = base64.b64encode(
                report.image_jpeg).decode("ascii")
        return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")

    def _body_event(self, event: Event) -> bytes | None:
        payload = getattr(event, "payload", None) or {}
        conf = payload.get("rer", payload.get("swing", 0.0))
        obj = {
            "type": "event",
            "camera_id": event.camera_id,
            "camera_name": self._camera_name.get(event.camera_id, event.camera_id),
            "timestamp": _now_iso(),
            "kind": event.kind,
            "algo": event.algo,
            "objects": [{"class": event.algo, "confidence": conf,
                         "bbox": _bbox(event)}],
            "metadata": payload,
        }
        return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


# =============================================================================
# Legacy shim：旧入口（server/worker.py）继续用 WEBHOOK_URL 环境变量。
# 新 Runtime 请改用 Reporter，不要新增对以下全局的依赖。
# =============================================================================
_URL = os.environ.get("WEBHOOK_URL", "").strip()
_COOLDOWN = float(os.environ.get("WEBHOOK_COOLDOWN_S", "60"))
_TIMEOUT = float(os.environ.get("WEBHOOK_TIMEOUT_S", "8"))

_q = queue.Queue(maxsize=64)
_last = {}
_lock = threading.Lock()
_started = False


def enabled() -> bool:
    return bool(_URL)


def _ensure():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_drain, daemon=True, name="webhook").start()


def _drain():
    while True:
        item = _q.get()
        if item is None:
            return
        url, body = item
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                logger.info("[webhook] {} -> {}", url, r.status)
        except Exception as e:
            logger.warning("[webhook] 上报失败: {}: {}", type(e).__name__, e)


def notify(camera_id, camera_name, events, jpeg_bytes, now_wall=None):
    """把 alarm 事件打包上报；任何异常都不得影响分析循环。"""
    if not _URL or not events:
        return
    try:
        _ensure()
        stamp = (now_wall or datetime.now().astimezone()).isoformat()
        for e in events:
            if e.kind != "alarm":
                continue
            key = (camera_id, e.algo, _target(e), e.kind)
            with _lock:
                if time.monotonic() - _last.get(key, 0.0) < _COOLDOWN:
                    continue
                _last[key] = time.monotonic()
            obj = {"class": e.algo,
                   "confidence": e.payload.get("rer", e.payload.get("swing", 0.0)),
                   "bbox": _bbox(e)}
            payload = {"camera_id": camera_id, "camera_name": camera_name,
                       "timestamp": stamp, "kind": e.kind, "algo": e.algo,
                       "objects": [obj], "metadata": e.payload}
            if jpeg_bytes:
                payload["frame_base64"] = base64.b64encode(jpeg_bytes).decode("ascii")
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            try:
                _q.put_nowait((_URL, body))
            except queue.Full:
                logger.warning("[webhook] 队列满，丢弃一条 alarm")
    except Exception as e:
        logger.warning("[webhook] notify 异常: {}: {}", type(e).__name__, e)
