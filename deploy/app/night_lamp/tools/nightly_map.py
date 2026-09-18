"""Nightly duty heatmap, steps 0-2: sample -> duty -> dimmed single color.

Step 0: coarse dynamic blobs (huge duty components: headlight wash) are
excluded from the P99 colormap input FIRST; classification never rescues a
scale that was already hijacked.
Fix 4: relative swing = peak excess / max(baseline, NOISE_FLOOR); the
reflection ratio test runs on rel, not absolute pixels (vignetting-proof).
Fix 5: reflection = ternary: xcorr(host)~1 AND rel-ratio<<1 AND host within
NEIGH_R px (~lamp housing scale; synced real lamps sit farther apart).
Dimming (dynamic blob / reflection) only darkens, never deletes.

No day overlay, no cross-night diff.
"""
import argparse
import json
import os
import sys

import cv2 as cv
import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from night_lamp.periodicity import (  # noqa: E402
    ac_limited, clean_train, onsets_of)

ON_DELTA = 40      # same delta as frozen lamp ON rule
BASELINE_N = 60    # evenly spaced frames for per-pixel night baseline
BLEND = 0.65       # heat weight over dimmed night frame
PCTL = 99.0        # colormap top = P99 of duty outside dynamic blobs
DYN_MIN_AREA = 2000  # duty component this big = headlight wash, not a lamp
CAND_DUTY_LO = 0.01  # candidate flashing pixel floor
NEIGH_R = 25       # host search radius, ~lamp housing px @1080p (tune per site)
SELF_R = 5         # exclusion disk around candidate when hunting host
XCORR_TH = 0.90
SWING_RATIO = 0.5
NOISE_FLOOR = 8.0  # dark-current-ish floor so rel never explodes on black
RENDER_DUTY_LO = 0.005  # soft floor: below this, paint fades to transparent
RENDER_GAMMA = 0.7  # <1 lifts mid duties; INFERNO's low end is too dark
DIM_DYNAMIC = 0.35  # dim, not delete: dynamic wash stays visible but grayed
DIM_REFLECT = 0.45
DYN_PEAK_FLOOR = 0.02  # local duty max inside a dyn blob = lamp core candidate
DYN_PEAK_WIN = 9       # local-max window (px) for dyn lamp cores
DYN_MAX_PEAK = 200     # cap on dyn lamp-core points sent to the periodicity pass
RING_COLOR = (0, 255, 0)  # BGR green ring: reads on day concrete and night
RING_THICK = 2            # ring stroke width (px)
RING_PAD = 3              # candidate ring radius = half bbox + pad
RING_R = 9                # dyn lamp-core ring radius (px)
DESPECKLE_FLOOR = 0.003  # shape floor: below this, a pixel is not a blob
DESPECKLE_K = 5     # neighborhood for the blob test
DESPECKLE_MIN = 3   # need this many neighbors (incl. self) to be painted
DAY_ALPHA = 0.9     # heat opacity over the day base
DAY_LO = 0.05       # day base: duty below this stays invisible
DAY_HI = 0.30       # day base: duty at/above this is fully painted
DAY_BG_SAT = 0.70   # day base: 1 = unchanged, 0 = grayscale
DAY_BG_DARK = 0.90  # day base brightness scale
CMAPS = {"inferno": cv.COLORMAP_INFERNO, "turbo": cv.COLORMAP_TURBO,
         "magma": cv.COLORMAP_MAGMA, "plasma": cv.COLORMAP_PLASMA,
         "jet": cv.COLORMAP_JET, "viridis": cv.COLORMAP_VIRIDIS,
         "ice": "ice"}
MAX_CAND = 600
PERIOD_STEP = 2     # stage-B fine sampling; ~3.5 samples per 0.56s flash
LAG_LO_S = 0.3      # AC window lower bound (seconds)
LAG_HI_S = 7.0      # >= MIN_PEAKS periods of the slowest lamp (T <= 2.2s)
PEAK_FLOOR = 0.2    # AC local max must clear this
MIN_PEAKS = 3       # clean train needs this many internal maxima
GAP_TOL = 1         # consecutive peak gaps must differ by <= this
ONSET_MIN = 50      # too few flashes -> trust duty, not periodicity
FLASH_R = 15        # disk painted around a flashed dyn sample
STEADY_LO = 0.5     # steady gate sigmoid center (duty)
STEADY_W = 0.08     # steady gate sigmoid width


def osd_mask(h, w):
    """Timestamp + 4G (top-right) and channel id (bottom-left)."""
    m = np.zeros((h, w), bool)
    m[0:int(h * 0.12), int(w * 0.62):] = True
    m[int(h * 0.90):, 0:int(w * 0.12)] = True
    return m


def roi_mask(h, w, calib_path):
    """Barrier-row polygons from water_barrier calibration (normalized)."""
    d = yaml.safe_load(open(calib_path, encoding="utf-8"))
    m = np.zeros((h, w), np.uint8)
    for r in d["rows"]:
        p = (np.asarray(r["poly"], dtype=np.float32) * [w, h]).astype(int)
        cv.fillPoly(m, [p], 255)
    return m > 0


def base_index(total, k):
    """K evenly spaced frame indices for baseline (seek pass)."""
    if total <= 0 or k <= 0:
        return []
    return [int(round(i * (total - 1) / max(k - 1, 1))) for i in range(min(k, total))]


def prep(gray, mask, off):
    """Mask raw frame first, then remove global offset. Order matters."""
    g = gray.copy()
    g[mask] = 0
    return g.astype(np.int16) - off


def med_series(video, step):
    """Pass 1 (sequential): per-sampled-frame global median series."""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("open failed: " + video)
    meds, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            meds.append(float(np.median(cv.cvtColor(fr, cv.COLOR_BGR2GRAY))))
        i += 1
    cap.release()
    if not meds:
        raise RuntimeError("no sampled frames: " + video)
    return np.array(meds, np.float32), i


def spread_gray(video, idxs):
    """Seek/read pass yielding grayscale frames at the given indices."""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("open failed: " + video)
    for j in idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, j)
        ok, fr = cap.read()
        if ok:
            yield cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
    cap.release()


def night_base(video, idxs, mask, detrend):
    """Per-pixel night baseline from K spread frames (masked + detrended)."""
    acc = []
    for g in spread_gray(video, idxs):
        off = int(round(float(np.median(g)) - detrend)) if detrend else 0
        acc.append(prep(g, mask, off))
    if not acc:
        raise RuntimeError("no baseline frames: " + video)
    return np.median(np.stack(acc), axis=0)


def bg_median(video, idxs):
    """Unmasked per-pixel median: render background without OSD black boxes."""
    acc = list(spread_gray(video, idxs))
    if not acc:
        raise RuntimeError("no background frames: " + video)
    return np.median(np.stack(acc), axis=0)


def duty_swing(video, base, step, meds, night_med, mask, delta):
    """Pass 3 (sequential): ON counts + peak excess per pixel, detrended."""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("open failed: " + video)
    on = np.zeros(base.shape, np.float32)
    peak = np.zeros(base.shape, np.float32)
    n = i = k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            v = prep(g, mask, int(round(meds[k] - night_med)))
            on += (v - base >= delta)
            np.maximum(peak, v - base, out=peak)
            n += 1
            k += 1
        i += 1
    cap.release()
    return on / n, peak, n


def split_comps(duty):
    """Connected flashing components -> dynamic blobs vs candidates.

    Gate on a duty floor, not duty>0: single-frame speckle (duty~1/N) would
    otherwise chain into one giant component and swallow the whole frame.
    """
    core = (duty >= CAND_DUTY_LO).astype(np.uint8)
    core = cv.morphologyEx(core, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, lab, stats, _ = cv.connectedComponentsWithStats(core, 8)
    dyn = np.zeros_like(duty, bool)
    cand = []
    for c in range(1, num):
        if stats[c, cv.CC_STAT_AREA] >= DYN_MIN_AREA:
            dyn |= lab == c
        else:
            cand.append(c)
    # Wash halo extends past the core: dilate so the whole glow dims, not a ring.
    dyn = cv.dilate(dyn.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    return dyn, lab, stats, cand


def label_slices(lab):
    """Single-pass ``{label: (ys, xs)}`` for every label present in ``lab``.

    Each pair is in the same row-major order as ``np.nonzero(lab == label)``,
    so callers that used a per-label full-frame scan get identical arrays at
    ``O(HW + nnz log nnz)`` total instead of ``O(n_labels * HW)``.
    """
    ys_all, xs_all = np.nonzero(lab)
    if ys_all.size == 0:
        return {}
    vals = lab[ys_all, xs_all]
    order = np.argsort(vals, kind="stable")
    ys_s, xs_s, vals_s = ys_all[order], xs_all[order], vals[order]
    uniq, starts, counts = np.unique(vals_s, return_index=True,
                                     return_counts=True)
    return {int(u): (ys_s[s:s + n], xs_s[s:s + n])
            for u, s, n in zip(uniq, starts, counts)}


def host_of(swing, y, x, r_out, r_in):
    """Brightest swing pixel in annulus (r_in, r_out]; None if flat."""
    h, w = swing.shape
    x0, x1 = max(0, x - r_out), min(w, x + r_out + 1)
    y0, y1 = max(0, y - r_out), min(h, y + r_out + 1)
    patch = swing[y0:y1, x0:x1].astype(np.float32).copy()
    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch[(yy - y) ** 2 + (xx - x) ** 2 <= r_in ** 2] = -1
    j = int(np.argmax(patch))
    if patch.flat[j] <= 0:
        return None
    return (y0 + j // patch.shape[1], x0 + j % patch.shape[1])


def series_for(video, step, pts, meds, night_med, mask):
    """Pass 4 (sequential): corrected gray series for P pixels only."""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("open failed: " + video)
    out = np.zeros((len(pts), len(meds)), np.int16)
    n = i = k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)
            off = int(round(meds[k] - night_med))
            for p, (y, x) in enumerate(pts):
                out[p, k] = int(g[y, x]) - (0 if mask[y, x] else off)
            n += 1
            k += 1
        i += 1
    cap.release()
    return out


def pearson(a, b):
    """Pearson r with flat-series guard (flat carries no signal)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def dim_map(duty, swing, base, dyn, lab, cand, series, r,
            dim_dyn=DIM_DYNAMIC, dim_ref=DIM_REFLECT):
    """Dimming only: dynamic blobs + ternary reflections go dark, kept."""
    dim = np.ones_like(duty, np.float32)
    dim[dyn] = dim_dyn
    rel = swing / np.maximum(base, NOISE_FLOOR)
    groups = label_slices(lab)
    n_ref = 0
    for t, c in enumerate(cand):
        ys, xs = groups.get(int(c), (None, None))
        if ys is None or ys.size == 0:
            continue
        j = int(np.argmax(swing[ys, xs]))
        oy, ox = int(ys[j]), int(xs[j])
        host = host_of(swing, oy, ox, r, SELF_R)
        if host is None:
            continue
        hy, hx = host
        if rel[hy, hx] <= 0:
            continue
        if (pearson(series[t], series[len(cand) + t]) >= XCORR_TH
                and rel[oy, ox] / rel[hy, hx] <= SWING_RATIO):
            dim[ys, xs] = dim_ref
            n_ref += 1
    return dim, n_ref, rel


def despeckle(duty, floor, k, min_n):
    """Keep pixels that are part of a blob, drop isolated ones.

    Speckle is single-pixel; a real (even faint) lamp is a blob. Counting
    neighbors is more targeted than raising the render floor, which would
    kill faint lamps along with the noise.
    """
    b = duty >= floor
    cnt = cv.boxFilter(b.astype(np.float32), -1, (k, k), normalize=False)
    return b & (cnt >= min_n)


def ice_lut():
    """Black -> teal -> cyan -> white; cold ramp, opposite the amber lamps."""
    stops = np.array([[0.00, 0, 0, 0],
                      [0.40, 128, 255, 0],
                      [0.75, 255, 255, 0],
                      [1.00, 255, 255, 255]], np.float32)
    x = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(x, stops[:, 0], stops[:, c + 1])
    return lut


ICE_LUT = ice_lut()


def heat_bgr(duty, top, gamma, cmap):
    """Colormap of (duty/top)^gamma; dim applied by the caller."""
    idx = (np.clip(duty / top, 0, 1) ** gamma * 255).astype(np.uint8)
    if isinstance(cmap, str):
        return ICE_LUT[idx].astype(np.float32)
    return cv.applyColorMap(idx, cmap).astype(np.float32)


def alpha_of(duty, keep, lo, scale):
    """Soft alpha: fade the duty floor to transparent, gate by despeckle."""
    a = keep * np.clip(duty / lo, 0, 1) * scale
    return np.repeat(a[..., None], 3, axis=2)


def alpha_ramp(duty, keep, lo, hi, scale):
    """Gated alpha for the day base: nothing below lo, full at hi.

    A bright base exposes thin low-duty wash (e.g. a headlight sweep spread
    over minutes) that the night base hides; only meaningful duty shows here.
    """
    a = keep * np.clip((duty - lo) / max(hi - lo, 1e-6), 0, 1) * scale
    return np.repeat(a[..., None], 3, axis=2)


def compose(heat, alpha, bg_bgr):
    return (heat * alpha + bg_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def night_bg(base):
    return cv.cvtColor((np.clip(base, 0, 255) * 0.35).astype(np.uint8),
                       cv.COLOR_GRAY2BGR)


def mute_bg(bgr, sat, dark):
    """Desaturate + darken a background so the heat ramp pops."""
    g3 = cv.cvtColor(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    out = bgr.astype(np.float32) * sat + g3.astype(np.float32) * (1 - sat)
    return np.clip(out * dark, 0, 255).astype(np.uint8)


def draw_rings(img, rings):
    """Outline each flashing lamp with a ring at its center."""
    for x, y, r in rings:
        cv.circle(img, (x, y), r, RING_COLOR, RING_THICK, cv.LINE_AA)
    return img


def draw_bar(img, cmap, title):
    """Vertical colorbar (low bottom, high top) with a title, top-right."""
    h, w = img.shape[:2]
    bw, bh = 16, int(h * 0.30)
    x0, y0 = w - bw - 24, int(h * 0.14)
    ramp = np.linspace(0, 255, bh).astype(np.uint8)[:, None]
    strip = np.repeat(ramp, bw, axis=1)
    bar = ICE_LUT[strip] if isinstance(cmap, str) else cv.applyColorMap(strip, cmap)
    bar = np.flipud(bar)
    cv.rectangle(img, (x0 - 1, y0 - 1), (x0 + bw, y0 + bh), (255, 255, 255), 1)
    img[y0:y0 + bh, x0:x0 + bw] = bar
    cv.putText(img, title, (x0 - 6 * len(title), y0 - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return img


def draw_caption(img, text):
    """Translucent dark strip + caption along the bottom edge."""
    h, w = img.shape[:2]
    band = img.copy()
    cv.rectangle(band, (0, h - 34), (w, h), (0, 0, 0), -1)
    out = cv.addWeighted(band, 0.5, img, 0.5, 0)
    cv.putText(out, text, (12, h - 11), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (255, 255, 255), 1, cv.LINE_AA)
    return out


def align_day(day_bgr, ref_gray):
    """ECC translation day->night; returns (aligned day, dx, dy, cc).

    Registration is a new single point of failure under this product shape:
    a wrong shift looks 'roughly right' to a human, so it is recorded in the
    ledger (dx/dy/cc), never silent.
    """
    h, w = ref_gray.shape
    if day_bgr.shape[:2] != (h, w):
        day_bgr = cv.resize(day_bgr, (w, h))
    dg = cv.cvtColor(day_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    rg = ref_gray.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        cc, warp = cv.findTransformECC(rg, dg, warp, cv.MOTION_TRANSLATION, crit)
    except cv.error:
        return day_bgr, 0.0, 0.0, -1.0
    out = cv.warpAffine(day_bgr, warp, (w, h),
                        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                        borderMode=cv.BORDER_REPLICATE)
    return out, float(warp[0, 2]), float(warp[1, 2]), float(cc)


def fine_meds(meds, step, fine_step, scanned):
    """Coarse global-median series resampled to the fine step (no re-median)."""
    src = np.arange(len(meds)) * step
    n_fine = (scanned + fine_step - 1) // fine_step
    return np.interp(np.arange(n_fine) * fine_step, src, meds).astype(np.float32)


def dyn_samples(duty, dyn, cap=DYN_MAX_PEAK):
    """Lamp-core points inside dynamic blobs: local duty maxima, not spread.

    A dense lamp chain (or a lamp inside its own glow) merges into one blob
    far larger than DYN_MIN_AREA, so uniform spread sampling misses most
    lamps. Local maxima track the lamps themselves, so every embedded
    flasher gets a fair periodicity test before the blob is written off as
    headlight wash.
    """
    d = np.where(dyn, duty, 0.0).astype(np.float32)
    mx = cv.dilate(d, np.ones((DYN_PEAK_WIN, DYN_PEAK_WIN), np.float32))
    peaks = (d >= mx) & (d >= DYN_PEAK_FLOOR) & dyn
    num, lab = cv.connectedComponents(peaks.astype(np.uint8), 8)
    groups = label_slices(lab)
    pts = []
    for c in range(1, num):
        ys, xs = groups.get(c, (None, None))
        if ys is None or ys.size == 0:
            continue
        j = int(np.argmax(d[ys, xs]))
        pts.append((int(ys[j]), int(xs[j]), float(d[ys[j], xs[j]])))
    pts.sort(key=lambda p: -p[2])   # strongest cores first when capped
    return [(y, x) for y, x, _ in pts[:cap]]


def steady_gate(duty, lo=STEADY_LO, w=STEADY_W):
    """Soft high-steady-duty gate: the OR branch that keeps steady lamps."""
    return (1.0 / (1.0 + np.exp((lo - duty) / w))).astype(np.float32)


def main():
    global ON_DELTA
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--delta", type=int, default=ON_DELTA)
    ap.add_argument("--baseline-n", type=int, default=BASELINE_N)
    ap.add_argument("--pctl", type=float, default=PCTL)
    ap.add_argument("--neigh-r", type=int, default=NEIGH_R)
    ap.add_argument("--gamma", type=float, default=RENDER_GAMMA)
    ap.add_argument("--render-lo", type=float, default=RENDER_DUTY_LO)
    ap.add_argument("--dim-dyn", type=float, default=DIM_DYNAMIC)
    ap.add_argument("--dim-ref", type=float, default=DIM_REFLECT)
    ap.add_argument("--desp-floor", type=float, default=DESPECKLE_FLOOR)
    ap.add_argument("--desp-k", type=int, default=DESPECKLE_K)
    ap.add_argument("--desp-min", type=int, default=DESPECKLE_MIN)
    ap.add_argument("--day", default=None, help="day image to overlay onto")
    ap.add_argument("--day-alpha", type=float, default=DAY_ALPHA)
    ap.add_argument("--day-lo", type=float, default=DAY_LO)
    ap.add_argument("--day-hi", type=float, default=DAY_HI)
    ap.add_argument("--cmap", default="ice", choices=sorted(CMAPS))
    ap.add_argument("--day-cmap", default="inferno", choices=sorted(CMAPS))
    ap.add_argument("--roi", default=None, help="water_barrier calib yaml")
    ap.add_argument("--no-detrend", action="store_true")
    a = ap.parse_args()
    ON_DELTA = a.delta

    cap = cv.VideoCapture(a.video)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 12.49)
    cap.release()
    mask = osd_mask(h, w)
    if a.roi:
        mask |= ~roi_mask(h, w, a.roi)

    meds, scanned = med_series(a.video, a.step)
    night_med = float(np.median(meds))
    detrend = 0.0 if a.no_detrend else night_med
    idxs = base_index(total, a.baseline_n)
    base = night_base(a.video, idxs, mask, detrend)
    bg = bg_median(a.video, idxs)
    duty, swing, n = duty_swing(a.video, base, a.step, meds, night_med,
                                mask, a.delta)

    # Step 0 first: P99 input excludes dynamic blobs AND the speckle floor,
    # else the scale is set by noise and every real lamp washes out.
    dyn, lab, stats, cand = split_comps(duty)
    rest = duty[(duty >= CAND_DUTY_LO) & ~dyn]
    if rest.size == 0:
        rest = duty[duty > 0]
    top = float(np.percentile(rest, a.pctl)) if rest.size else 1.0
    top = top if top > 0 else 1.0

    # Fix 5 needs time series, but only for small candidates.
    cand = sorted(cand, key=lambda c: -stats[c, cv.CC_STAT_AREA])[:MAX_CAND]
    pts, hosts = [], []
    for c in cand:
        ys, xs = np.nonzero(lab == c)
        j = int(np.argmax(swing[ys, xs]))
        oy, ox = int(ys[j]), int(xs[j])
        host = host_of(swing, oy, ox, a.neigh_r, SELF_R)
        pts.append((oy, ox))
        hosts.append(host or (oy, ox))
    series = series_for(a.video, a.step, pts + hosts, meds, night_med, mask)
    dim, n_ref, rel = dim_map(duty, swing, base, dyn, lab, cand, series,
                              a.neigh_r, a.dim_dyn, a.dim_ref)

    # Stage B: periodicity, candidates only. A clean AC train (equal spacing)
    # = flasher; monotone decay = one-off transit; near-steady high duty is
    # caught by the steady gate instead. Only dims/reflections stay artifacts.
    # Test the max-DUTY pixel (lamp core), not max-swing: the swing peak sits
    # on the glow edge where the baseline is already high, so a real flasher
    # reads as flat there and gets dropped.
    fpts = []
    for c in cand:
        ys, xs = np.nonzero(lab == c)
        j = int(np.argmax(duty[ys, xs]))
        fpts.append((int(ys[j]), int(xs[j])))
    dyn_pts = dyn_samples(duty, dyn)
    sb = series_for(a.video, PERIOD_STEP, fpts + dyn_pts,
                    fine_meds(meds, a.step, PERIOD_STEP, scanned),
                    night_med, mask)
    lag_lo = max(1, int(round(LAG_LO_S * fps / PERIOD_STEP)))
    lag_hi = int(round(LAG_HI_S * fps / PERIOD_STEP))

    def flashing(i, y, x):
        on = (sb[i].astype(np.float64) - base[y, x]) >= a.delta
        if onsets_of(on) < ONSET_MIN:
            return False
        curve = ac_limited(on, lag_lo, lag_hi)[2]
        return clean_train(curve, lag_lo, PEAK_FLOOR, MIN_PEAKS, GAP_TOL)[0]

    rings = []
    flash = np.zeros((h, w), np.uint8)
    n_flash = 0
    for i, c in enumerate(cand):
        if flashing(i, *fpts[i]):
            flash[lab == c] = 1
            r = max(stats[c, cv.CC_STAT_WIDTH],
                    stats[c, cv.CC_STAT_HEIGHT]) // 2 + RING_PAD
            rings.append((fpts[i][1], fpts[i][0], r))
            n_flash += 1
    for j, (y, x) in enumerate(dyn_pts):
        if flashing(len(cand) + j, y, x):
            cv.circle(flash, (x, y), FLASH_R, 1, -1)
            rings.append((x, y, RING_R))
            n_flash += 1
    flash = flash > 0

    ref = dim == a.dim_ref
    dim[flash & ~ref] = 1.0            # a rescued flasher is not an artifact
    gate = np.maximum(flash.astype(np.float32), steady_gate(duty))

    keep = despeckle(duty, a.desp_floor, a.desp_k, a.desp_min)
    heat = heat_bgr(duty, top, a.gamma, CMAPS[a.cmap]) * dim[..., None]
    vis = compose(heat,
                  alpha_of(duty, keep, a.render_lo, BLEND)
                  * (dim * gate)[..., None],
                  night_bg(bg))
    draw_rings(vis, rings)
    draw_bar(vis, CMAPS[a.cmap], "flash duty")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv.imwrite(a.out, vis)

    shift = None
    if a.day:
        day = cv.imread(a.day)
        if day is None:
            raise RuntimeError("day image not found: " + a.day)
        day, dx, dy, cc = align_day(day, np.clip(bg, 0, 255))
        dheat = heat_bgr(duty, top, a.gamma, CMAPS[a.day_cmap]) * dim[..., None]
        dalpha = (alpha_ramp(duty, keep, a.day_lo, a.day_hi, a.day_alpha)
                  * (dim * gate)[..., None])
        dvis = compose(dheat, dalpha, mute_bg(day, DAY_BG_SAT, DAY_BG_DARK))
        draw_rings(dvis, rings)
        draw_bar(dvis, CMAPS[a.day_cmap], "flash duty")
        dvis = draw_caption(
            dvis, "night %s  duty>=%.2f  delta=%d  p99=%.3f"
            % (os.path.basename(a.video), a.day_lo, a.delta, top))
        dout = os.path.splitext(a.out)[0] + "_day.jpg"
        cv.imwrite(dout, dvis)
        shift = {"day": a.day, "dx": round(dx, 2), "dy": round(dy, 2),
                 "ecc": round(cc, 4), "out": dout}
        print("  day overlay: %s (dx=%.2f dy=%.2f ecc=%.4f)" %
              (dout, dx, dy, cc))

    np.save(os.path.splitext(a.out)[0] + "_duty.npy", duty.astype(np.float32))
    np.save(os.path.splitext(a.out)[0] + "_rel.npy",
            rel.astype(np.float32))
    json.dump({"video": a.video, "frames_total": scanned, "frames_used": n,
               "step": a.step, "delta": a.delta, "baseline_n": a.baseline_n,
               "detrend": not a.no_detrend,
               "global_med": round(night_med, 1),
               "global_std": round(float(meds.std()), 2),
               "pctl": a.pctl, "duty_pctl": round(top, 4),
               "dyn_frac": round(float(dyn.mean()), 4),
               "n_cand": len(cand), "n_reflect": n_ref, "n_flash": n_flash,
               "neigh_r": a.neigh_r, "period_step": PERIOD_STEP,
               "lag_frames": [lag_lo, lag_hi], "onset_min": ONSET_MIN,
               "peak_floor": PEAK_FLOOR, "gap_tol": GAP_TOL,
               "min_peaks": MIN_PEAKS, "steady_lo": STEADY_LO,
               "gamma": a.gamma, "render_lo": a.render_lo,
               "dim_dyn": a.dim_dyn, "dim_ref": a.dim_ref,
               "desp_floor": a.desp_floor, "desp_k": a.desp_k,
               "desp_min": a.desp_min, "cmap": a.cmap,
               "desp_kept": round(float(keep.mean()), 5),
               "rel_max": round(float(rel.max()), 2),
               "duty_max": round(float(duty.max()), 4),
               "duty_mean": round(float(duty.mean()), 6),
               "day": shift, "roi": a.roi},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print("saved %s (used %d, p99=%.3f dyn=%.3f cand=%d reflect=%d "
          "flash=%d keep=%.4f)" %
          (a.out, n, top, dyn.mean(), len(cand), n_ref, n_flash,
           keep.mean()))


if __name__ == "__main__":
    main()
