"""离线夜灯跑批：整段夜间录像 -> 逐时累计热力图（供人工检查）。

用生产同款 ``NightAdapter`` + ``FakeClock`` 按真实时间轴驱动 ``NightSession``：

* burst 节奏与线上一致（``overnight.cadence_minutes`` / ``burst_seconds``）；
* 额外每 ``--snap-every-min`` 分钟产出一张累计热力图，便于看整夜演进；
* 收尾 ``freeze``+``finalize`` 产出整夜最终图（有 ``--day`` 时叠加白天底图）。

输出（默认 ``output/night_test/<date>/<scheme>/``）::

    snap_<NN>_<视频时刻>.jpg/.json    周期累计热力图
    burst_<NN>_<视频时刻>.jpg/.json   线上节奏的整点快照
    final_night.jpg / final_day.jpg   最终整夜热力图
    final.json / meta.json            最终元数据 / 本次运行摘要
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
APP = TOOLS.parent / "app"
REPO = TOOLS.parent.parent
sys.path.insert(0, str(APP))

import cv2 as cv  # noqa: E402

from night_lamp.adapter import NightAdapter  # noqa: E402
from night_lamp.session import NightSession  # noqa: E402
from server import config  # noqa: E402
from server.schedule import FakeClock  # noqa: E402

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / 1048576.0


def _series_mb(session) -> float:
    s = getattr(session, "_series", None)
    return 0.0 if s is None else s.memory_bytes() / 1048576.0

_TZ = timezone(timedelta(hours=8))
_META_KEYS = (
    "alignment_status", "day_alignment", "image_mode", "n_cand", "n_cand_total",
    "n_flash", "n_reflect", "candidate_overflow", "night_state",
    "night_qualified", "night_gate", "memory_estimate_mb", "warmup",
    "delta", "lag_frames", "period_step", "restored", "reason",
)
_FINAL_KEYS = ("alignment_status", "day_alignment", "image_mode", "n_cand",
               "n_cand_total", "n_flash", "n_reflect", "candidate_overflow",
               "night_state", "night_qualified", "memory_estimate_mb", "warmup")


def _auto_day(camera: str) -> Path | None:
    d = REPO / "data" / camera
    if not d.is_dir():
        return None
    jpgs = sorted(d.glob("*day.jpg"))
    return jpgs[-1] if jpgs else None


def _stamp(clock: FakeClock) -> str:
    return clock.wall().strftime("%H%M%S")


def _save_report(rep, out_dir: Path, prefix: str, idx: int, clock: FakeClock,
                 full_meta: bool) -> dict | None:
    if rep is None:
        return None
    stamp = _stamp(clock)
    stem = "%s_%02d_%s" % (prefix, idx, stamp)
    image_name = None
    if rep.image_jpeg:
        (out_dir / (stem + ".jpg")).write_bytes(rep.image_jpeg)
        image_name = stem + ".jpg"
    md = rep.metadata or {}
    keys = _META_KEYS if full_meta else _FINAL_KEYS
    meta = {
        "created_at": rep.created_at,
        "status": rep.status,
        "report_type": rep.report_type,
        "image": image_name,
        "metadata": {k: md.get(k) for k in keys if k in md},
    }
    (out_dir / (stem + ".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return meta


def run(args) -> dict:
    cfg = config.load_runtime(Path(args.config) if args.config else None)
    cam = next(c for c in cfg.cameras if c.id == args.camera)
    spec = cam.algorithms["night_lamp"].spec

    # 参数覆盖（用于对照实验 / 找优化组合）
    if args.continuous:
        spec.overnight.enabled = False
    if args.interval_ms:
        spec.sampling.interval_ms = int(args.interval_ms)
    if args.burst_seconds:
        spec.overnight.burst_seconds = int(args.burst_seconds)
    if args.cadence_minutes:
        spec.overnight.cadence_minutes = int(args.cadence_minutes)
    if args.max_candidates:
        spec.memory.max_candidates = int(args.max_candidates)
    if args.series_cap is not None:
        spec.memory.series_cap = (None if args.series_cap == 0
                                  else int(args.series_cap))

    video = Path(args.video)
    if not video.is_absolute():
        video = REPO / video
    cap = cv.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    if args.fps:
        fps = float(args.fps)
    if fps <= 1e-6:
        fps = 12.5
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    dt = 1.0 / fps

    start = datetime.fromisoformat(args.start) if args.start else \
        datetime(2026, 9, 21, 20, 0, 0, tzinfo=_TZ)
    if start.tzinfo is None:
        start = start.replace(tzinfo=_TZ)

    day = Path(args.day) if args.day else _auto_day(cam.id)
    base = cv.imread(str(day)) if day else None

    session = NightSession(cam.id, spec)
    clock = FakeClock(start)
    adapter = NightAdapter(session, spec, clock)

    out_dir = Path(args.out) if args.out else (REPO / "output" / "night_test")
    out_dir = out_dir / args.scheme
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_every = max(int(args.snap_every_min), 0) * 60.0
    # 半周期偏移，避免周期快照正好撞上 burst 边界把 burst 快照挤掉
    next_snap = (snap_every / 2.0) if snap_every > 0 else None
    n_snap = n_burst = 0
    frames = 0
    t0 = time.perf_counter()
    perf_every = max(int(args.perf_every), 1)
    perf = []
    rss_start = _rss_mb()
    rss_peak = rss_start
    rss_prev = rss_start

    print("video=%s fps=%.3f frames=%s dur=%.1fmin camera=%s scheme=%s"
          % (video.name, fps, total or "?", (total / fps / 60.0) if total else 0,
             cam.id, args.scheme))
    print("out=%s day=%s overnight=%s"
          % (out_dir, day.name if day else None, spec.overnight))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        clock.advance(dt)
        adapter.on_frame(frame, clock.wall(), clock.monotonic())

        for rep in adapter.take_reports():
            n_burst += 1
            _save_report(rep, out_dir, "burst", n_burst, clock, full_meta=True)

        if next_snap is not None and clock.monotonic() >= next_snap:
            next_snap += snap_every
            try:
                rep = session.snapshot(clock.wall())
            except Exception as exc:  # noqa: BLE001
                print("  [snap warn] %s: %s" % (type(exc).__name__, exc))
                rep = None
            saved = _save_report(rep, out_dir, "snap", n_snap + 1, clock,
                                 full_meta=True)
            if saved is not None:
                n_snap += 1

        if frames % perf_every == 0:
            now = time.perf_counter()
            rss = _rss_mb()
            rss_peak = max(rss_peak, rss)
            perf.append({
                "frame": frames,
                "video_s": round(clock.monotonic(), 1),
                "samples": int(session.sample_count),
                "rss_mb": round(rss, 1),
                "rss_delta_mb": round(rss - rss_prev, 1),
                "session_mb": round(_series_mb(session), 2),
                "elapsed_s": round(now - t0, 1),
                "ms_per_sample": round((now - t0) * 1000.0
                                       / max(session.sample_count, 1), 2),
            })
            rss_prev = rss
        if args.max_frames and frames >= args.max_frames:
            print("  [max-frames] stop at %d" % frames)
            break
        if frames % 5000 == 0:
            print("  frames=%d t=%.0fs samples=%d bursts=%d snaps=%d rss=%.0fMB"
                  % (frames, clock.monotonic(), session.sample_count,
                     n_burst, n_snap, _rss_mb()))

    cap.release()
    adapter.freeze()
    t_fin = time.perf_counter()
    # finalize 只读累计状态、不 freeze/release，故可调两次分别出夜图与白天叠加图。
    rep_night = session.finalize(None, clock.wall())
    rep_day = session.finalize(base, clock.wall()) if base is not None else None
    fin_s = time.perf_counter() - t_fin

    if rep_night is not None and rep_night.image_jpeg:
        (out_dir / "final_night.jpg").write_bytes(rep_night.image_jpeg)
    if rep_day is not None and rep_day.image_jpeg:
        (out_dir / "final_day.jpg").write_bytes(rep_day.image_jpeg)

    primary = rep_day if rep_day is not None else rep_night
    final_meta = None
    if primary is not None:
        md = primary.metadata or {}
        final_meta = {
            "created_at": primary.created_at,
            "status": primary.status,
            "report_type": primary.report_type,
            "images": [n for n in ("final_night.jpg", "final_day.jpg")
                       if (out_dir / n).exists()],
            "metadata": {k: md.get(k) for k in _META_KEYS if k in md},
            "adapter": adapter.stats(),
        }
        (out_dir / "final.json").write_text(
            json.dumps(final_meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    rss_end = _rss_mb()
    perf_summary = {
        "rss_start_mb": round(rss_start, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "rss_end_mb": round(rss_end, 1),
        "rss_growth_mb": round(rss_peak - rss_start, 1),
        "session_end_mb": round(_series_mb(session), 2),
        "samples": int(session.sample_count),
        "wall_s": round(time.perf_counter() - t0, 1),
        "samples_per_s": round(session.sample_count
                               / max(time.perf_counter() - t0, 1e-6), 2),
        "finalize_s": round(fin_s, 1),
        "max_candidates": int(spec.memory.max_candidates),
        "series_cap": spec.memory.series_cap,
        "interval_ms": int(spec.sampling.interval_ms),
    }
    (out_dir / "perf.json").write_text(
        json.dumps({"summary": perf_summary, "series": perf},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    summary = {
        "video": str(video),
        "scheme": args.scheme,
        "camera": cam.id,
        "fps": fps,
        "frames_read": frames,
        "video_seconds": round(clock.monotonic(), 1),
        "samples": int(session.sample_count),
        "snapshots": n_snap,
        "bursts": n_burst,
        "wall_elapsed_s": round(time.perf_counter() - t0, 1),
        "finalize_s": round(fin_s, 1),
        "out_dir": str(out_dir),
        "day_base": str(day) if day else None,
        "overnight": {"enabled": bool(spec.overnight.enabled),
                      "cadence_minutes": spec.overnight.cadence_minutes,
                      "burst_seconds": spec.overnight.burst_seconds},
        "adapter_stats": adapter.stats(),
        "perf": perf_summary,
        "final": final_meta,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("done: frames=%d video=%.0fs samples=%d bursts=%d snaps=%d "
          "finalize=%.1fs status=%s"
          % (frames, clock.monotonic(), session.sample_count, n_burst, n_snap,
             fin_s, getattr(primary, "status", None)))
    print("out -> %s" % out_dir)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--camera", default="1749")
    ap.add_argument("--config", default=None)
    ap.add_argument("--day", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", default=None, help="ISO 起始墙钟（默认 2026-09-21T20:00+08:00）")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--snap-every-min", type=int, default=10)
    ap.add_argument("--perf-every", type=int, default=2000)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--continuous", action="store_true",
                    help="关闭 overnight burst，整段连续累计（省 CPU 前请权衡内存）")
    ap.add_argument("--interval-ms", type=int, default=0)
    ap.add_argument("--burst-seconds", type=int, default=0)
    ap.add_argument("--cadence-minutes", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--series-cap", type=int, default=None,
                    help="0=整段全序列(None)，>0=只保留末尾 N 个样本")
    a = ap.parse_args()
    if not a.scheme:
        a.scheme = Path(a.video).stem
    run(a)


if __name__ == "__main__":
    main()
