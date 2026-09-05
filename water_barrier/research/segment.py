"""分割基座：把"水马"（红底座 + 白板）从白天实流里稳健地割出来。

相比第一版纯"红∪白"色阈值的改进：
  1. 红底座锚定：红色（色相红 + S + a*）几乎不会出现在路面/树影/灰车上，是最稳的信号。
     先用红底座定位水马，再只保留"靠近红底座的白板"，从而弃掉远处高亮但不属于水马的
     白色路面、白色货车、天空等地物。白板若不贴近红底座则视为误检，剔除。
  2. 近严远宽：远端带色相不稳、S 低，放宽阈值，但仍以红底座锚定。
  3. 形态学：开运算去噪 + 闭运算 + 沿路向线段核把断缝抹平，连成屏障带。
  4. 连通域稳定化（'largest + second-largest' 思想）：屏障带可能被缺口/遮挡切成多块，
     只保留面积前三大连通块（含红或白校验），丢弃零星噪声块，使后续缺口检测不被杂块干扰。

阈值来源：data 已标定版本（1622x905），按帧分辨率等比缩放；硬编码以便独立复现。
"""
import cv2 as cv
import numpy as np

CALIB_W, CALIB_H = 1622, 905

# --- 色彩阈值 ---
RED_H1, RED_H2, RED_H3, RED_H4 = 0, 10, 168, 180
RED_S_MIN, RED_A_MIN = 60, 145
WHITE_V_MIN, WHITE_S_MAX = 195, 60
FAR_RED_S_MIN, FAR_RED_A_MIN, FAR_RED_V_MIN = 25, 134, 140
# 连通域均值色体检（防灰路面/树影混入）：偏红 或 偏白 二选一
CC_A_REDISH, CC_V_WHITEISH, CC_S_WHITEISH = 136, 190, 80

# --- 形态学核 ---
OPEN_K, CLOSE_K, KNIT_H, KNIT_V = (3, 3), (9, 9), (1, 21), (21, 1)
# 红底座锚定小间隙：白板紧跟红底座即同一水马单元，缝隙极小，用该距离把两者连到一个 CC
RED_ANCHOR_SIZE = (9, 9)
MIN_CC_AREA, FAR_MIN_CC_AREA = 12, 6
# 连通域稳定化：保留面积前 TOP_CC 的大块（等价"最大 + 次大"），忽略更小的噪声块
TOP_CC = 3


def kernels(w, h):
    sx, sy = w / CALIB_W, h / CALIB_H
    k = lambda kw, kh: (max(1, round(kw * sx)), max(1, round(kh * sy)))
    a = lambda v: max(1, round(v * sx * sy))
    return (k(*OPEN_K), k(*CLOSE_K), k(*KNIT_H), k(*KNIT_V),
            k(*RED_ANCHOR_SIZE), a(MIN_CC_AREA), a(FAR_MIN_CC_AREA))


def rect_mask(shape, rects, val=255):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = val
    return m


def _color_masks(hsv, lab, s_red, a_red, v_red, v_white, s_white):
    """一组（红,白）掩膜；远端版多一个 V 下限防暗色噪声。"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = cv.inRange(h, RED_H1, RED_H2) | cv.inRange(h, RED_H3, RED_H4)
    red = hue & cv.inRange(s, s_red, 255) & cv.inRange(lab[..., 1], a_red, 255)
    if v_red:
        red &= cv.inRange(v, v_red, 255)
    white = cv.inRange(v, v_white, 255) & cv.inRange(s, 0, s_white)
    return red, white


def _cc_mean(hsv, lab, labels, i, x, y, w, h):
    m = labels[y:y + h, x:x + w] == i
    return (lab[..., 1][y:y + h, x:x + w][m].mean(),
            hsv[..., 2][y:y + h, x:x + w][m].mean(),
            hsv[..., 1][y:y + h, x:x + w][m].mean())  # a*, V, S


def _keep_big_components(fg, far_band, min_cc, min_cc_far, hsv, lab, max_keep=TOP_CC):
    """连通域清杂 + 稳定化：剔除小斑、颜色既非红也非白的、以及面积掉出前 max_keep 的。"""
    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)

    # 过滤1/2：面积 + 均值色体检
    kept_idx = []
    for i in range(1, n):
        x, y, w, h, area = stats[i, :5]
        if area < (min_cc_far if far_band[y + h // 2, x + w // 2] else min_cc):
            continue
        a, v, s = _cc_mean(hsv, lab, labels, i, x, y, w, h)
        if a >= CC_A_REDISH or (v >= CC_V_WHITEISH and s <= CC_S_WHITEISH):
            kept_idx.append(i)

    # 过滤3：只保留面积最大的 max_keep 个（"最大 + 次大"思想），丢弃零星噪声块。
    if not kept_idx:
        return np.zeros_like(fg)
    kept_idx.sort(key=lambda i: stats[i, cv.CC_STAT_AREA], reverse=True)
    kept_idx = kept_idx[:max_keep]

    out = np.zeros_like(fg)
    for i in kept_idx:
        out[labels == i] = 255
    return out


def rois(shape):
    return rect_mask(shape, ((0, 330, 900, 930), (1040, 295, 1462, 395), (880, 370, 1468, 950)))


def far_band_of(shape):
    return rect_mask(shape, ((1040, 295, 1462, 395), (1040, 375, 1170, 445)))


def exclude_mask(shape):
    return rect_mask(shape, ((1100, 25, 1595, 135), (15, 825, 125, 900),
                             (1272, 280, 1318, 306), (1420, 272, 1458, 306)))


def _white_near_red(white, red, gap):
    """保留"紧贴红底座"的白板连通块：整块白板跟它的红底座同属一个水马单元。

    用 CC 相交而非距离阈值：白板与（膨胀小间隙后的）红底座有交集的连通块整体保留，
    从而整块大尺寸近景白板都能收进来；与之无连接的孤立白色（路面/白车/天空）丢弃。
    尺度无关，远近景通用。
    """
    red_d = cv.dilate(red, cv.getStructuringElement(cv.MORPH_RECT, gap))
    n, labels = cv.connectedComponents(white, 8)
    out = np.zeros_like(white)
    for i in range(1, n):
        m = labels == i
        if np.any(red_d[m]):
            out[m] = 255
    return out


def segment(frame: np.ndarray, roi=None, far_band=None) -> np.ndarray:
    """帧 -> 稳健水马 mask（0/255，同尺寸）。

    roi / far_band 为可选几何约束（None 则用内置粗 ROI）。仅当几何约束缺失时使用内置。
    返回的 mask 由"红底座锚定"的屏障带 + 前 max_keep 大连通块构成。
    """
    h, w = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    if roi is None:
        roi = rois((h, w))
    if far_band is None:
        far_band = far_band_of((h, w)) & roi
    else:
        far_band = far_band & roi

    k_open, k_close, k_knit_h, k_knit_v, k_anchor, min_cc, min_cc_far = kernels(w, h)

    red, white = _color_masks(hsv, lab, RED_S_MIN, RED_A_MIN, 0,
                              WHITE_V_MIN, WHITE_S_MAX)
    red_f, white_f = _color_masks(hsv, lab, FAR_RED_S_MIN, FAR_RED_A_MIN,
                                  FAR_RED_V_MIN, WHITE_V_MIN, WHITE_S_MAX)
    for m in (red, white, red_f, white_f):
        m[:] = cv.morphologyEx(m, cv.MORPH_OPEN, np.ones(k_open, np.uint8))

    # --- 红底座锚定：整块白板必须与红底座相连，否则弃（去高亮路面/白车/天空） ---
    white = _white_near_red(white, red | red_f, k_anchor)
    white_f = _white_near_red(white_f, red | red_f, k_anchor)

    # 近严远宽再并；远端白已锚定到（近端∪远端）红底座
    fg = (red | white) | ((red_f | white_f) & far_band)

    # 形态学连片
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_close, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_h, np.uint8))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones(k_knit_v, np.uint8))
    fg &= (~exclude_mask((h, w))) & roi

    # 连通域稳定化
    return _keep_big_components(fg, far_band, min_cc, min_cc_far, hsv, lab)
