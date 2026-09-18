"""单帧信号链（纯函数，无跨帧状态）：采样 -> 质量门 -> 平滑/包络/成核 -> 候选。

三段合一（原 sampling.py / detector.py / quality.py / alarm.py 单帧部分）：
  采样  沿水马轴向切站（M 原生站），重采样到 REF 归一化轴；
  质量  解码损坏判整帧 INVALID，异物/暗段判该行 OCCLUDED；
  检测  红白信号平滑 + 包络基线 + floor/宽度成核 + U 宽度门。
跨帧累计与时序确认见 track.py，装配见 pipeline.py。
"""
import cv2
import numpy as np

# ---------------------------------------------------------------- 采样
# 轴向采样步长 2.0 像素/站；截面有效至少 MINH=10 像素才计站。
STEP = 2.0
MINH = 10

# REF 轴： барьер轴重采样站数（与图像分辨率无关的归一化坐标；support/成框全在此轴上）。
REF = 812

# 成核门限（顶层标定直通，见 pipeline.DETECT_KEYS）：
# floor_max 缺口内信号下限；core_exit core 边缘迟滞；min_width 包络最小宽；
# core_min_width core 最小宽（单位均为 REF 站信号值/站数）。
FLOOR_MAX = 0.18
CORE_EXIT = 0.06
MIN_WIDTH = 40
CORE_MIN_WIDTH = 30
# 平滑/基线窗口（REF 站）：SMOOTH_SIGMA 高斯 σ；BASE_WIN 包络均值/膨胀窗。
SMOOTH_SIGMA = 6
BASE_WIN = 251

# ---------------------------------------------------------------- 质量门
# 解码损坏门：32x32 块梯度均值的最大块（PMAX_TH=2.0，BADFRAME-001 实测，超限整帧 INVALID）。
PMAX_TH = 2.0

# 遮挡门（REF 站上连续跑长）：异色像素占比>FOREIGN_TH 连 FOREIGN_SPAN 站，
# 或信号暗于 DARK_TH 连 DARK_SPAN 站，即判该行 OCCLUDED（OCCLUSION-001）。
FOREIGN_TH = 0.3
FOREIGN_SPAN = 60
DARK_TH = 0.05
DARK_SPAN = 200


def _row_axis(poly_norm):
    c = poly_norm.mean(axis=0)
    _, _, vt = np.linalg.svd(poly_norm - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d[0] < 0:
        d = -d
    rel = (poly_norm - c) @ d
    return {"d": d.astype(np.float32), "c": c.astype(np.float32),
            "umin": float(rel.min()), "umax": float(rel.max())}


def _pack(col, foreign, valid_station, m, umin, step, d, c):
    def rs(a, interp):
        return cv2.resize(a.reshape(1, -1), (REF, 1), interpolation=interp).ravel()

    v = rs(valid_station.astype(np.uint8), cv2.INTER_NEAREST) > 0
    v = cv2.erode(v.astype(np.uint8).reshape(1, -1), np.ones((1, 7))).ravel() > 0
    return {"s": rs(col.astype(np.float32), cv2.INTER_LINEAR),
            "f": rs(foreign.astype(np.float32), cv2.INTER_LINEAR),
            "v": v, "M": m, "umin": umin, "step": step, "d": d, "c": c}


def _resample_y(ym, cnt):
    """Per-M-station y-extent -> REF arrays. Gaps filled by index interp (measured)."""
    ym = ym.astype(np.float32)
    ym[cnt == 0] = np.nan
    idx = np.arange(len(ym))
    ok = np.isfinite(ym)
    if not ok.any():
        return np.full(REF, np.nan, np.float32)
    filled = np.interp(idx, idx[ok], ym[ok]).astype(np.float32)
    return cv2.resize(filled.reshape(1, -1), (REF, 1), interpolation=cv2.INTER_LINEAR).ravel()


def sample_row(bgr, poly_norm, step_px=STEP, minh=MINH):
    """Sample along barrier axis. Section split matches E5: top 60% white, bottom 40% red."""
    h, w = bgr.shape[:2]
    pts = poly_norm * np.array([w, h], dtype=np.float32)
    ax = _row_axis(poly_norm)
    d = ax["d"] * np.array([w, h], dtype=np.float32)
    d = d / np.linalg.norm(d)
    c = ax["c"] * np.array([w, h], dtype=np.float32)
    rel = (pts - c) @ d
    umin, umax = float(rel.min()), float(rel.max())
    m = max(1, int(np.ceil((umax - umin) / step_px)))
    x0, y0 = pts.min(axis=0).astype(int).clip(0)
    x1, y1 = pts.max(axis=0).astype(int)
    x1, y1 = min(w - 1, int(x1)), min(h - 1, int(y1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(pts - np.array([x0, y0])).astype(np.int32)], 1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        z = np.zeros(m, np.float32)
        pack = _pack(z, z, np.zeros(m, bool), m, umin, step_px, d, c)
        pack["y0"] = np.full(REF, np.nan, np.float32)
        pack["y1"] = np.full(REF, np.nan, np.float32)
        return pack

    col_id = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
    station = np.floor(((col_id - c) @ d - umin) / step_px).astype(int).clip(0, m - 1)

    rm = red_m(bgr)
    wm = white_m(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32)
    is_red_hue = (hue <= 12) | (hue >= 165)
    foreign = ((sat > 100) & (~is_red_hue)).astype(np.float32)

    red_px = rm[ys + y0, xs + x0].astype(np.float32)
    white_px = wm[ys + y0, xs + x0].astype(np.float32)
    foreign_px = foreign[ys + y0, xs + x0]
    rows_y = (ys + y0).astype(np.float32)

    order = np.lexsort((rows_y, station))
    ss = station[order]
    cnt = np.bincount(ss, minlength=m)
    pos = np.arange(len(ss)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    is_white = pos < (0.6 * cnt[ss]).astype(int)

    white_sum = np.bincount(ss[is_white], weights=white_px[order][is_white], minlength=m)
    white_cnt = np.bincount(ss[is_white], minlength=m)
    red_sum = np.bincount(ss[~is_white], weights=red_px[order][~is_white], minlength=m)
    red_cnt = np.bincount(ss[~is_white], minlength=m)
    foreign_sum = np.bincount(ss, weights=foreign_px[order], minlength=m)

    valid_station = cnt >= minh
    col = np.zeros(m, np.float32)
    foreign_col = np.zeros(m, np.float32)
    ok = valid_station & (white_cnt > 0) & (red_cnt > 0)
    col[ok] = (0.6 * red_sum[ok] / red_cnt[ok] + 0.4 * white_sum[ok] / white_cnt[ok]).astype(np.float32)
    foreign_col[valid_station] = (foreign_sum[valid_station] / cnt[valid_station]).astype(np.float32)

    abs_y = (ys + y0).astype(np.float32)
    ymin_m = np.full(m, np.inf, np.float32)
    ymax_m = np.full(m, -np.inf, np.float32)
    np.minimum.at(ymin_m, station, abs_y)
    np.maximum.at(ymax_m, station, abs_y)

    pack = _pack(col, foreign_col, valid_station, m, umin, step_px, d, c)
    pack["y0"] = _resample_y(ymin_m, cnt)
    pack["y1"] = _resample_y(ymax_m, cnt)
    return pack


def apply_end_trim(pack, frac):
    """Mark both axis ends invalid: section collapses at polygon vertices."""
    v = pack["v"].copy()
    k = int(REF * frac)
    v[:k] = False
    v[REF - k:] = False
    v = cv2.erode(v.astype(np.uint8).reshape(1, -1), np.ones((1, 7))).ravel() > 0
    return {**pack, "v": v}


def sta_to_xy(pack, sta812, w, h):
    """Station (812 coords) -> image point, for overlay only."""
    s = (np.asarray(sta812, dtype=np.float32) + 0.5) / REF * pack["M"]
    pts = pack["c"] + np.outer(pack["umin"] + (s + 0.5) * pack["step"], pack["d"])
    pts[:, 0] = pts[:, 0].clip(0, w - 1)
    pts[:, 1] = pts[:, 1].clip(0, h - 1)
    return pts


def red_m(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    r = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    r = r | cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    return r > 0


def white_m(bgr):
    """水马白（含阴影白）：日光白 S<=70&V>=150；阴影白 S 70~120&V 80~150&H>=85。

    阴影白依据 2026-09-10 实测：阴影水马 HSV~(90,80,110)，与晴天路面 (90,20,170)
    在 S 上差 4 倍（天空光偏冷 vs 日光直射）。H>=85 剔绿叶（H~60~75），路面 H 无意义
    （低 S）故不受影响。阴影白计入 white 后采样信号与 RER 前景同步生效。
    残留风险：红棕泥土（H 小）仍会落入；S>120 的深影白仍走旧逻辑。回归矩阵看守。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sunlit = (s <= 70) & (v >= 150)
    shadow = (s > 70) & (s <= 120) & (v >= 80) & (v < 150) & (h >= 85)
    return sunlit | shadow


def smooth(s, v):
    ss = s.copy()
    if (~v).any() and v.any():
        idx = np.arange(len(s))
        ss[~v] = np.interp(idx[~v], idx[v], s[v])
    k = SMOOTH_SIGMA * 6 + 1
    return cv2.GaussianBlur(ss.reshape(1, -1), (k, 1), SMOOTH_SIGMA).ravel()


def _masked_mean(sm, v):
    num = cv2.blur((sm * v).reshape(1, -1), (BASE_WIN, 1)).ravel()
    den = cv2.blur(v.astype(np.float32).reshape(1, -1), (BASE_WIN, 1)).ravel()
    return num / np.maximum(den, 1e-6)


def mbase(sm, v):
    mean = _masked_mean(sm, v)
    tmp = sm.copy().astype(np.float32)
    tmp[~v] = -np.inf
    with np.errstate(invalid="ignore"):
        mx = cv2.dilate(tmp.reshape(1, -1), np.ones((1, BASE_WIN))).ravel()
    mx[~v] = np.nan
    return np.maximum(np.nan_to_num(mean), np.nan_to_num(mx))


def detect(sm, bs, v, floor_max=FLOOR_MAX, core_exit=CORE_EXIT,
           min_width=MIN_WIDTH, core_min_width=CORE_MIN_WIDTH):
    """Envelope discovery d>0.25, extend d>0.125; core is S<floor_max run."""
    d = bs - sm
    outs, i, n = [], 0, len(sm)
    while i < n:
        if (not v[i]) or d[i] <= 0.25:
            i += 1
            continue
        j = i
        while j < n and v[j] and d[j] > 0.125:
            j += 1
        if v[i:j].sum() / max(1, j - i) < 0.9 or (j - i) < min_width:
            i = j
            continue
        env = (int(i), int(j - 1))
        k, best = i, None
        while k < j:
            if sm[k] >= floor_max:
                k += 1
                continue
            m = k
            while m < j and sm[m] < floor_max + core_exit:
                m += 1
            e = m
            while e - 1 >= k and sm[e - 1] >= floor_max:
                e -= 1
            if e - k >= core_min_width:
                best = (int(k), int(e - 1), int(e - k), round(float(sm[k:e].min()), 3))
                break
            k = m
        outs.append({"env": env, "core": best, "prom": round(float(d[i:j].max()), 3),
                     "reason": "FLOOR+WIDTH" if best else "FLOOR_FAIL"})
        i = j
    return outs


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


def confidence(prom, alen_stations, unit):
    """宽度门（硬）+ 突出度。侧翼门已删：87 分钟 1106 个 core 只翻转 3 次结论
    （audit.py，2026-09-10），留着只是装饰复杂度。"""
    if unit > 0 and (alen_stations < 0.5 * unit or alen_stations > 2.0 * unit):
        return 0.0
    return round(min(1.0, max(0.0, prom)), 3)


def detect_frame(bgr, polys, row_ids, units, trim, conf_th,
                 floor_max, core_exit, min_width, core_min_width):
    """单帧 -> (kept, skip, packs)。

    kept {行id: [(a, b)]} REF 站坐标 core 区间；packs {行id: pack} 供成框复用，
    避免对同一帧重复采样（确定性函数，复用与重采数值一致）。
    """
    if len(polys) != len(row_ids):
        raise ValueError(f"polys行数({len(polys)})与row_ids({len(row_ids)})不一致")
    ok, _ = frame_valid(bgr)
    if not ok:
        return {}, {"all": "INVALID"}, {}
    kept, skip, packs = {}, {}, {}
    for rid, poly in zip(row_ids, polys):
        if rid not in units:
            raise ValueError(f"行缺U标定: {rid}")
        pack = sample_row(bgr, poly)
        packs[rid] = pack
        valid = apply_end_trim(pack, trim)["v"]
        if is_occluded(pack, valid):
            kept[rid] = []
            skip[rid] = "OCCLUDED"
            continue
        unit = units[rid]
        sm = smooth(pack["s"], valid)
        row = []
        for c in detect(sm, mbase(sm, valid), valid, floor_max=floor_max,
                        core_exit=core_exit, min_width=min_width,
                        core_min_width=core_min_width):
            if not (c["reason"] == "FLOOR+WIDTH" and c["core"]):
                continue
            a, b = c["core"][0], c["core"][1]
            if confidence(c["prom"], (b - a) / REF * pack["M"], unit) >= conf_th:
                row.append((a, b))
        kept[rid] = row
    return kept, skip, packs


def pixel_box(pack, a, b, w, h):
    """REF 站区间 -> 像素框（实测轴 + 截面 y 范围，仅用于输出标注）。"""
    pts = sta_to_xy(pack, [a, b], w, h)
    x0, x1 = int(round(pts[0][0])), int(round(pts[1][0]))
    top = pack["y0"][a:b + 1]
    bot = pack["y1"][a:b + 1]
    if np.isfinite(top).any() and np.isfinite(bot).any():
        y0, y1 = int(np.nanmin(top)), int(np.nanmax(bot))
    else:
        cy = int(round((pts[0][1] + pts[1][1]) / 2))
        y0, y1 = cy - 20, cy + 20
    x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
    return (x0, y0, x1, y1)


def find_runs(sup, thresh, min_width):
    """support 一维数组 -> [(a, b)] REF 站区间（闭区间）。"""
    boxes, i = [], 0
    while i < REF:
        if sup[i] < thresh:
            i += 1
            continue
        j = i
        while j < REF and sup[j] >= thresh:
            j += 1
        if j - i >= min_width:
            boxes.append((int(i), int(j - 1)))
        i = j
    return boxes
