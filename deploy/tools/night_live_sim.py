"""用录像模拟实时流，验证「墙钟分桶」第一阶段：

* 桶按墙钟整点开/结算，与是否有帧无关（断流也出图）；
* on_frame 只喂数据，同一 frame_id 不重复计入；
* 注入：抖动 / 短断流 / 整桶断流 / 进程重启；
* 校验：整点出图不丢不重、silent_bucket、coverage 正确、重启不重复 publish。

用法（仓库根）::

    <venv>/python deploy/tools/night_live_sim.py \
        --video tmp/2026-09-21/1002490_20260921_cont60.mp4 \
        --out output/live_sim --horizon-s 8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
sys.path.insert(0, str(APP))

import cv2 as cv  # noqa: E402

from night_lamp.adapter import NightAdapter  # noqa: E402
from night_lamp.persist import NightStateStore  # noqa: E402
from night_lamp.session import NightSession  # noqa: E402
from server import config  # noqa: E402
from server.schedule import FakeClock  # noqa: E402

_TZ = timezone(timedelta(hours=8))


def _build(cam, spec, clock, store):
    ad = NightAdapter(NightSession(cam.id, spec), spec, clock)
    if store is not None:
        ad.attach_persistence(store)
        ad.restore()
    return ad


def run(args) -> dict:
    cfg = config.load_runtime(Path(args.config) if args.config else None)
    cam = next(c for c in cfg.cameras if c.id == args.camera)
    spec = cam.algorithms["night_lamp"].spec
    if args.interval_ms:
        spec.sampling.interval_ms = int(args.interval_ms)
    if args.burst_seconds:
        spec.overnight.burst_seconds = int(args.burst_seconds)
    burst = int(spec.overnight.burst_seconds)

    video = Path(args.video)
    if not video.is_absolute():
        video = REPO / video
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.5)
    dt = 1.0 / fps

    start = datetime(2026, 9, 21, 20, 0, 0, tzinfo=_TZ)
    clock = FakeClock(start)
    store_root = tempfile.mkdtemp(prefix="night_live_sim_")
    store = NightStateStore(store_root)

    out = Path(args.out) if args.out else (REPO / "output" / "live_sim")
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    jitter = float(args.jitter)
    horizon = float(args.horizon_s)
    # 桶窗口：每整点前 burst 秒；注入的断流区间与重启时刻（模拟时间秒）
    windows = [(k * 3600.0, k * 3600.0 + burst) for k in range(0, 8)]
    if args.no_faults:
        outages, restarts = [], []
    else:
        outages = [(150.0, 210.0), (3600.0, 3600.0 + burst)]  # 短断 + 整桶断
        restarts = [7500.0, 8000.0]                           # 窗内重启 + 出图后重启

    adapter = _build(cam, spec, clock, store)
    reader = iter_video(cap)
    seq = 0
    reported = []       # (bucket, report_dict)

    def drain():
        for rep in adapter.take_reports():
            md = rep.metadata
            stamp = (rep.created_at or "unknown").replace(":", "").replace("-", "")
            stem = "bucket_%s" % (md.get("bucket") or stamp)
            if rep.image_jpeg:
                (out / (stem + ".jpg")).write_bytes(rep.image_jpeg)
            rec = {
                "bucket": md.get("bucket"),
                "created_at": rep.created_at,
                "status": rep.status,
                "image": (stem + ".jpg") if rep.image_jpeg else None,
                "scheduled_steps": md.get("scheduled_steps"),
                "actual_frames": md.get("actual_frames"),
                "unique_frames": md.get("unique_frames"),
                "coverage": md.get("coverage"),
                "silent_bucket": md.get("silent_bucket"),
                "n_cand": md.get("n_cand"),
                "n_cand_total": md.get("n_cand_total"),
                "n_flash": md.get("n_flash"),
            }
            (out / (stem + ".json")).write_text(
                json.dumps(rec, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            reported.append(rec)

    t0 = time.perf_counter()
    restarts_done = []
    t = clock.monotonic()
    while t < horizon:
        in_win = any(a <= t < b for a, b in windows)
        in_outage = any(a <= t < b for a, b in outages)
        if in_win and not in_outage:
            frame = reader()
            if frame is None:
                break
            seq += 1
            d = dt * (1.0 + rng.uniform(-jitter, jitter)) if jitter else dt
            clock.advance(d)
            adapter.on_frame(frame, clock.wall(), clock.monotonic(),
                             frame_id=seq)
        else:
            # 非窗口/断流：不喂数据，仅推进墙钟（并 tick 以触发桶边界）
            if in_win:
                clock.advance(dt)
            else:
                nxt = min([a for a, _ in windows if a > t] or [horizon])
                clock.advance(min(30.0, max(1e-3, nxt - t)))
        adapter.tick(clock.wall())
        t = clock.monotonic()
        drain()
        for rt in restarts:
            if rt not in restarts_done and t >= rt:
                restarts_done.append(rt)
                try:
                    adapter.save_state(clock.wall())   # 模拟生产 flush_night
                except Exception:
                    pass
                adapter.release()
                adapter = _build(cam, spec, clock, store)
    drain()
    if args.finalize:
        adapter.freeze()
        adapter.finalize(None, clock.wall())
        drain()

    # ---- 校验 ----
    buckets = [r["bucket"] for r in reported]
    expected = []
    for a, b in windows:
        if a + burst <= horizon:
            expected.append((start + timedelta(seconds=a)).strftime("%Y-%m-%dT%H"))
    problems = []
    for e in expected:
        if buckets.count(e) != 1:
            problems.append("bucket %s emitted %d time(s)" % (e, buckets.count(e)))
    silent = [r["bucket"] for r in reported if r["silent_bucket"]]
    cross = (start + timedelta(seconds=3600)).strftime("%Y-%m-%dT%H")
    if not args.no_faults and cross not in silent:
        problems.append("cross-bucket outage bucket %s not silent" % cross)
    for r in reported:
        if not r["silent_bucket"] and not (0.0 < (r["coverage"] or 0) <= 1.0):
            problems.append("coverage out of range for %s: %s"
                            % (r["bucket"], r["coverage"]))

    summary = {
        "video": str(video),
        "burst_seconds": burst,
        "horizon_s": horizon,
        "jitter": jitter,
        "outages": outages,
        "restarts": restarts_done,
        "expected_buckets": expected,
        "reported": reported,
        "silent_buckets": silent,
        "wall_elapsed_s": round(time.perf_counter() - t0, 1),
        "problems": problems,
        "ok": not problems,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n=== buckets ===")
    for r in reported:
        print("  %s  cover=%s silent=%s frames=%s/%s cand=%s flash=%s"
              % (r["bucket"], r["coverage"], r["silent_bucket"],
                 r["unique_frames"], r["actual_frames"], r["n_cand_total"],
                 r["n_flash"]))
    print("restarts=%s wall=%.0fs" % (restarts_done, summary["wall_elapsed_s"]))
    print("PROBLEMS:", problems if problems else "none")
    print("out -> %s" % out)
    cap.release()
    return summary


def iter_video(cap):
    def _next():
        ok, frame = cap.read()
        return frame if ok else None
    return _next


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--horizon-s", type=float, default=8000.0)
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--burst-seconds", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-faults", action="store_true",
                    help="关闭抖动/断流/重启注入，用于与离线结果做等价对照")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    r = run(a)
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
