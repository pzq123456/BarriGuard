"""Evidence writers -- EXTRACTED from overnight_run.py, math unchanged.

Snapshot semantics (frozen): raw + overlay JPEG, rep-frame picked by global
median proximity, quality from config. Pure visualization, never feeds detector.
"""
import json
import os

import cv2 as cv
import numpy as np


def rep_idx(frame_meds):
    """Representative frame: global median closest to burst median."""
    meds = np.array(frame_meds)
    return int(np.argmin(np.abs(meds - np.median(meds))))


def snap_paths(out_dir, tag):
    d = os.path.join(out_dir, "snapshots")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, tag + ".raw.jpg"), os.path.join(d, tag + ".overlay.jpg")


def save_snap(out_dir, bgr, tag, lamps, cands, quality=70):
    """Save raw + overlay. Returns (raw_rel, overlay_rel) relative to out_dir."""
    raw_p, ovl_p = snap_paths(out_dir, tag)
    cv.imwrite(raw_p, bgr, [cv.IMWRITE_JPEG_QUALITY, quality])
    vis = bgr.copy()
    for l in lamps:  # frozen registry dots + ids, color by controls
        col = {"positive": (255, 255, 0),
               "steady_check": (0, 255, 255)}.get(l.get("control"), (0, 255, 0))
        cv.circle(vis, (int(l["x"]), int(l["y"])), 7, col, 2)
        cv.putText(vis, l["id"], (int(l["x"]) + 9, int(l["y"]) - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for c in cands or []:  # candidate diamonds + ids (watchlist / east propose)
        cv.drawMarker(vis, (int(c["x"]), int(c["y"])), (255, 0, 255),
                      cv.MARKER_DIAMOND, 16, 2)
        cv.putText(vis, c.get("id", "?"), (int(c["x"]) + 9, int(c["y"]) + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    cv.imwrite(ovl_p, vis, [cv.IMWRITE_JPEG_QUALITY, quality])
    return os.path.relpath(raw_p, out_dir), os.path.relpath(ovl_p, out_dir)


class Jsonl:
    """Append-only jsonl handles (one per stream). Caller closes."""

    def __init__(self, out_dir, mode="w"):
        self._fhs = {}
        self._out = out_dir
        self._mode = mode

    def open(self, *names):
        for n in names:
            self._fhs[n] = open(os.path.join(self._out, n),
                                self._mode, encoding="utf-8")
        return self

    def write(self, name, obj):
        self._fhs[name].write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush(self):
        for fh in self._fhs.values():
            fh.flush()

    def close(self):
        for fh in self._fhs.values():
            fh.close()
        self._fhs = {}
