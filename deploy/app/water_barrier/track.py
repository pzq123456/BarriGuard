import cv2
import numpy as np

from .signal import (CORE_EXIT, CORE_MIN_WIDTH, FLOOR_MAX, MIN_WIDTH, REF,
                     detect_frame, find_runs, pixel_box)

NORMAL, SUSPECTED, ALARM = "NORMAL", "SUSPECTED", "ALARM"

# support 累计 fallback（标定 yaml 里已显式写出同样值；缺键时才用此处）。
DEFAULT_TRIM = 0.04
DEFAULT_CONF_TH = 0.30
DEFAULT_SUPPORT_TH = 0.3
DEFAULT_MIN_BOX_WIDTH = 30
DEFAULT_FLOOR_MAX = FLOOR_MAX
DEFAULT_CORE_EXIT = CORE_EXIT
DEFAULT_MIN_WIDTH = MIN_WIDTH
DEFAULT_CORE_MIN_WIDTH = CORE_MIN_WIDTH


def fit_road_profile(road_roi_bgr, patch_size=8):
    """路面 Lab 分布（在 patch 均值上拟合）-> profile dict。

    用 patch 均值而非像素级：像素级 sigma 夸大纹理噪声、错置 q95 门限，
    在 1333 帧上把真缺口 RER 从 0.47 拉低到 0.27。
    """
    if road_roi_bgr.size == 0:
        raise ValueError("路面色标定框为空。")
    lab = cv2.cvtColor(road_roi_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    samples = _patch_means(lab, patch_size)
    mean_l = float(np.mean(samples[:, 0]))
    std_l = float(np.std(samples[:, 0]))
    mean_ab = np.mean(samples[:, 1:], axis=0)
    std_ab = np.maximum(np.std(samples[:, 1:], axis=0), 0.1)
    dist = np.sqrt(np.sum(((samples[:, 1:] - mean_ab) / std_ab) ** 2, axis=1))
    return {"mean_l": mean_l, "mean_ab": mean_ab, "std_ab": std_ab,
            "l_max": min(255.0, mean_l + 2.5 * std_l),
            "dist_th": float(np.percentile(dist, 95)) + 0.5}


def _patch_means(lab_roi, patch_size):
    h, w, _ = lab_roi.shape
    ny, nx = h // patch_size, w // patch_size
    if ny == 0 or nx == 0:
        return lab_roi.reshape(-1, 3)
    grid = lab_roi[: ny * patch_size, : nx * patch_size]
    return grid.reshape(ny, patch_size, nx, patch_size, 3).swapaxes(1, 2).reshape(
        ny * nx, patch_size, patch_size, 3).mean(axis=(1, 2))


def patch_distance(profile, patch_mean_lab):
    if patch_mean_lab[0] > profile["l_max"]:
        return 999.0
    norm = (patch_mean_lab[1:] - profile["mean_ab"]) / profile["std_ab"]
    return float(np.sqrt(np.sum(norm ** 2)))


def calc_patch_rer(image_bgr, slot_mask, foreground_mask, road_profile,
                   purity=0.70, ps=8):
    """缺口区域"路露率"：缺口未被水马掩膜覆盖的区域中，真实路面 patch 占比。

    非路内容（货车/行人/杂物）RER<=0.10，真移除缺口 RER 0.27-0.65，阈值取 0.25。
    """
    missing = cv2.bitwise_and(slot_mask, cv2.bitwise_not(foreground_mask))
    if cv2.countNonZero(missing) < 200:
        return 0.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    h, w, _ = lab.shape
    ny, nx = h // ps, w // ps
    cropped_lab = lab[: ny * ps, : nx * ps]
    cropped_mask = missing[: ny * ps, : nx * ps]
    lp = cropped_lab.reshape(ny, ps, nx, ps, 3).swapaxes(1, 2)
    mp = cropped_mask.reshape(ny, ps, nx, ps).swapaxes(1, 2)
    counts = np.count_nonzero(mp, axis=(2, 3))
    valid = counts >= purity * ps * ps
    if not np.any(valid):
        return 0.0
    patch_means = lp[valid].mean(axis=(1, 2))
    road = sum(1 for m in patch_means
               if patch_distance(road_profile, m) <= road_profile["dist_th"])
    return float(road) / float(patch_means.shape[0])


def new_track(box, tcfg):
    """新建轨道 dict（阈值由标定注入，此处不硬编码）。"""
    return {"box": box, "severity": 0.0, "row_id": "", "last_seen": 0.0,
            "decay": tcfg["decay_rate"], "hold": tcfg["alarm_hold_s"],
            "reconfirm": tcfg["reconfirm_s"], "rer_th": tcfg["rer_threshold"],
            "reset": tcfg["intact_reset_s"], "cap": tcfg["median_window"],
            "state": NORMAL, "accum": 0.0, "last_update": -1.0,
            "frames": [], "rer_t": None, "rer": 0.0,
            "rer_below": 0, "intact_sec": 0.0}


def track_push_frame(tr, frame_bgr):
    tr["frames"].append(frame_bgr)
    if len(tr["frames"]) > tr["cap"]:
        tr["frames"].pop(0)


def track_median(tr):
    if not tr["frames"]:
        return None
    return np.median(np.stack(tr["frames"], axis=0), axis=0).astype(np.uint8)


def track_update(tr, defective, now, confirm_fn):
    """推进状态机一步，返回 state（"NORMAL"/"SUSPECTED"/"ALARM"）。

    defective=True 缺检测器报出该缺口；False 视为完好。
    confirm_fn(中值帧)->RER，持续到 hold 后才调用（<=reconfirm 节流一次），
    迟滞 2 次才降级。
    """
    dt = max(0.0, now - tr["last_update"]) if tr["last_update"] >= 0.0 else 0.0
    tr["last_update"] = now

    if defective:
        tr["accum"] += dt
        tr["intact_sec"] = 0.0
    else:
        tr["accum"] = max(0.0, tr["accum"] - dt * tr["decay"])
        tr["intact_sec"] += dt
        if tr["intact_sec"] >= tr["reset"]:
            tr["accum"] = 0.0
            tr["rer_t"], tr["rer"] = None, 0.0
            tr["rer_below"] = 0
            tr["state"] = NORMAL
            return tr["state"]

    if tr["accum"] <= 0.0:
        tr["state"] = NORMAL
        tr["rer_below"] = 0
        return tr["state"]

    if tr["accum"] < tr["hold"]:
        tr["state"] = SUSPECTED
        return tr["state"]

    if tr["rer_t"] is None or now - tr["rer_t"] >= tr["reconfirm"]:
        med = track_median(tr)
        tr["rer"] = confirm_fn(med) if med is not None else 0.0
        tr["rer_t"] = now
    tr["rer_below"] += 1
    if tr["rer"] >= tr["rer_th"]:
        tr["state"] = ALARM
        tr["rer_below"] = 0
    elif tr["rer_below"] >= 2:
        tr["state"] = SUSPECTED
    return tr["state"]


def new_support_state(polys, row_ids, units, trim=DEFAULT_TRIM,
                      conf_th=DEFAULT_CONF_TH, support_th=DEFAULT_SUPPORT_TH,
                      min_box_width=DEFAULT_MIN_BOX_WIDTH,
                      floor_max=DEFAULT_FLOOR_MAX, core_exit=DEFAULT_CORE_EXIT,
                      min_width=DEFAULT_MIN_WIDTH,
                      core_min_width=DEFAULT_CORE_MIN_WIDTH):
    """新建 support 累计状态（dict，调用方持有；多流各一份）。"""
    if len(polys) != len(row_ids):
        raise ValueError(f"polys行数({len(polys)})与row_ids({len(row_ids)})不一致")
    for rid in row_ids:
        if rid not in units:
            raise ValueError(f"行缺U标定: {rid}")
    return {"polys": list(polys), "row_ids": list(row_ids),
            "units": dict(units),
            "trim": trim, "conf_th": conf_th,
            "support_th": support_th, "min_box_width": min_box_width,
            "floor_max": floor_max, "core_exit": core_exit,
            "min_width": min_width, "core_min_width": core_min_width,
            "acc": {rid: np.zeros(REF, np.float32) for rid in row_ids},
            "n_obs": {rid: 0 for rid in row_ids},
            "alarms": []}


def reset_support(st):
    """清 support 累计（多流生命周期管理用，平时不调）。"""
    for rid in st["row_ids"]:
        st["acc"][rid] = np.zeros(REF, np.float32)
        st["n_obs"][rid] = 0
    st["alarms"] = []


def support_of(st):
    """当前各行 support（累计命中/有效帧数），仅供调试。"""
    return {rid: st["acc"][rid] / max(1, st["n_obs"][rid]) for rid in st["row_ids"]}


def step_support(st, bgr):
    """喂一帧，返回当前报警 [(row_id, sta, box, support)]。

    INVALID/OCCLUDED 帧不计入累计；质量门失败时冻结上次输出（不清零）。
    """
    if bgr is None:
        return list(st["alarms"])
    h, w = bgr.shape[:2]
    kept, skip, packs = detect_frame(
        bgr, st["polys"], st["row_ids"], st["units"], st["trim"], st["conf_th"],
        st["floor_max"], st["core_exit"], st["min_width"], st["core_min_width"])
    if "all" in skip:
        return list(st["alarms"])
    for rid in st["row_ids"]:
        if rid in skip:
            continue
        st["n_obs"][rid] += 1
        for a, b in kept.get(rid, []):
            st["acc"][rid][a:b + 1] += 1
    sup = support_of(st)
    alarms = []
    for rid in st["row_ids"]:
        pack = packs[rid]
        for a, b in find_runs(sup[rid], st["support_th"], st["min_box_width"]):
            alarms.append((rid, (a, b), pixel_box(pack, a, b, w, h),
                           round(float(sup[rid][a:b + 1].max()), 3)))
    st["alarms"] = alarms
    return list(st["alarms"])
