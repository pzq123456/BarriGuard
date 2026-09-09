"""Deterministic file replay -- MOVED verbatim from stream_source.FileSrc.

Home of FileSrc (production camera.py imports from here, never duplicates it).
CLI: dump a burst or single frame for inspection / regression.

  python tools/replay.py --video ../tmp/1749_202609040100.mp4 --f0 0 --n 40 --out burst_preview.jpg
"""
import argparse
import os
import sys

import cv2 as cv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class FileSrc:
    """Deterministic file source: burst_at(f0, n) -> n BGR frames."""

    def __init__(self, path):
        self.path = path
        self._cap = cv.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError("打不开视频: " + str(path))
        self._fps = float(self._cap.get(cv.CAP_PROP_FPS))
        self._total = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))

    def fps(self):
        return self._fps

    def total(self):
        return self._total

    def burst_at(self, f0, n):
        self._cap.set(cv.CAP_PROP_POS_FRAMES, f0)
        out = []
        for _ in range(n):
            ok, fr = self._cap.read()
            if not ok:
                break
            out.append(fr)
        if len(out) != n:
            raise RuntimeError("short_read f0=%s got=%s/%s" % (f0, len(out), n))
        return out

    def frame_at(self, f0):
        """Single frame (mid-interval snapshot use)."""
        self._cap.set(cv.CAP_PROP_POS_FRAMES, f0)
        ok, fr = self._cap.read()
        if not ok:
            raise RuntimeError("seek_read_fail f0=%s" % f0)
        return fr

    def close(self):
        self._cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    src = FileSrc(args.video)
    print("fps=%s total=%s" % (src.fps(), src.total()), flush=True)
    frames = src.burst_at(args.f0, args.n)
    print("got %s frames" % len(frames), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        cv.imwrite(args.out, frames[len(frames) // 2])
        print("wrote " + args.out, flush=True)
    src.close()


if __name__ == "__main__":
    main()
