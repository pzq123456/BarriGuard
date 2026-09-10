"""Frame quality gates: decode corruption + occlusion.

Kept from BADFRAME-001 / OCCLUSION-001. Single-frame row_state() is dropped:
gapwatch only needs the maxrun primitive plus the frozen thresholds.
"""

import cv2
import numpy as np

PMAX_TH = 2.0

FOREIGN_TH = 0.3
FOREIGN_SPAN = 60
DARK_TH = 0.05
DARK_SPAN = 200


def frame_valid(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = mag.shape
    ph, pw = h // 32, w // 32
    pmax = float(mag[:ph * 32, :pw * 32].reshape(ph, 32, pw, 32).mean(axis=(1, 3)).max())
    return pmax < PMAX_TH, round(pmax, 3)


def maxrun(mask):
    best, i = 0, 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        best = max(best, j - i)
        i = j
    return best


def is_occluded(pack, valid):
    foreign_run = maxrun(valid & (pack["f"] > FOREIGN_TH)) > FOREIGN_SPAN
    dark_run = maxrun(valid & (pack["s"] < DARK_TH)) > DARK_SPAN
    return foreign_run or dark_run
