"""Decoder-side (FFmpeg) cost benchmark -- Experiment C.

Measures the real decode cost of the 10-min night clip under different
decoder-side filters, WITHOUT going through the Python application.  This makes
explicit that the production ``interval_ms`` only drops frames *after* decode:
the rawvideo pipe in ``deploy/app/server/source.py`` decodes every frame.

Variants
--------
* full        : decode native resolution, no filter
* fps=6.25    : decimate to the production application sampling rate
* scale=960   : decode then downscale to 960x540
* scale=640   : decode then downscale to 640x360
* scale+fps   : downscale + decimate

Substream is not testable offline (no second RTSP track in a file); note it.

Usage
-----
    python benchmark/benchmark_ffmpeg.py --video tmp/1002490_night_10min.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

BENCH_RE = re.compile(r"bench:\s*utime=([\d.]+)s\s*stime=([\d.]+)s\s*rtime=([\d.]+)s")

VARIANTS = {
    "full": [],
    "fps6.25": ["-vf", "fps=6.25"],
    "scale1280": ["-vf", "scale=1280:-2"],
    "scale960": ["-vf", "scale=960:-2"],
    "scale640": ["-vf", "scale=640:-2"],
    "scale960_fps6.25": ["-vf", "scale=960:-2,fps=6.25"],
}


def run_once(ffmpeg: str, video: Path, vf: list[str], timeout: float) -> dict:
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-benchmark",
           "-i", str(video), *vf, "-f", "null", "-"]
    t = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          timeout=timeout)
    wall = time.perf_counter() - t
    err = proc.stderr.decode(errors="replace")
    m = BENCH_RE.search(err)
    if not m:
        return {"wall_s": round(wall, 3), "utime_s": None, "stime_s": None,
                "rtime_s": None, "rc": proc.returncode}
    return {"wall_s": round(wall, 3),
            "utime_s": float(m.group(1)), "stime_s": float(m.group(2)),
            "rtime_s": float(m.group(3)), "rc": proc.returncode}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="tmp/1002490_night_10min.mp4")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="benchmark/out")
    a = ap.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found on PATH; cannot run Experiment C")
        return 2
    video = Path(a.video)
    if not video.is_file():
        raise FileNotFoundError(video)

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, vf in VARIANTS.items():
        runs = [run_once(ffmpeg, video, vf, a.timeout) for _ in range(max(1, a.reps))]
        good = [r for r in runs if r["utime_s"] is not None]
        best = min(good, key=lambda r: r["utime_s"]) if good else runs[0]
        results[name] = {"vf": " ".join(vf), "best": best, "runs": runs}
        print(f"{name:18s} utime={best['utime_s']} stime={best['stime_s']} "
              f"rtime={best['rtime_s']} wall={best['wall_s']}s")

    fp = outdir / f"ffmpeg_{time.strftime('%Y%m%d_%H%M%S')}.json"
    fp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
