"""确定性验证告警帧落盘链路（不需要真实缺口）。

Reporter -> 本地 webhook_receiver -> 磁盘 jpg/json。
另外验证 ReportStore 落盘 Report。
"""
from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
OUT = REPO / "output"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(REPO / "local"))  # webhook_receiver

import cv2 as cv  # noqa: E402

import webhook_receiver as wr  # noqa: E402
from server import config  # noqa: E402
from server.algo import Event, Report  # noqa: E402
from server.output import ReportStore  # noqa: E402
from server.reporting import Reporter  # noqa: E402

PORT = 9999
FRAME = REPO / "data" / "1749" / "1749_20260917_155840_day.jpg"


def main():
    alert_dir = OUT / "alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    wr._save_dir = alert_dir
    wr._payload_log = alert_dir / "payload.jsonl"

    httpd = HTTPServer(("127.0.0.1", PORT), wr.WebhookHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("receiver up on 127.0.0.1:%d -> %s" % (PORT, alert_dir))

    cfg = config.load_runtime(TOOLS.parent / "config.yaml")
    cfg.callback.url = "http://127.0.0.1:%d/alert" % PORT
    cfg.callback.enabled = True

    jpeg = FRAME.read_bytes()
    event = Event(camera_id="1749", algo="water_gap", ts=time.time(), kind="alarm",
                  payload={"row_id": "row_2_right_near", "severity": 51,
                           "rer": 0.62, "box": [1100, 500, 1200, 600]})
    reporter = Reporter(cfg)
    reporter.submit_event(event, image_jpeg=jpeg)
    time.sleep(2.0)
    reporter.close()

    # Report store
    store = ReportStore(str(OUT / "reports"))
    rep = Report(camera="1749", algorithm="water_gap", report_type="gap_overlay",
                 created_at=datetime.now(timezone.utc).isoformat(),
                 image_jpeg=jpeg, status="ok",
                 metadata={"rows": ["row_2_right_near"], "status": "ALARM"})
    store.save(rep)

    httpd.shutdown()
    time.sleep(0.3)

    alerts = sorted(alert_dir.rglob("*.jpg"))
    metas = sorted(alert_dir.rglob("*.json"))
    reports = sorted((OUT / "reports").rglob("*"))
    print("ALERT frames saved: %d" % len(alerts))
    for p in alerts:
        print("  ", p.relative_to(REPO))
    print("ALERT meta saved: %d" % len(metas))
    print("REPORT files saved: %d" % len([p for p in reports if p.is_file()]))
    for p in reports:
        if p.is_file():
            print("  ", p.relative_to(REPO))
    ok = bool(alerts) and bool(metas) and any(p.suffix == ".jpg" for p in reports)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
