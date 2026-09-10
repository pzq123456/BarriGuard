"""缺口时序确认与状态机：由 tmp/server/gap_detector.py 移植，去掉固定 slot 概念。

原 slot 版以固定窗口的"覆盖率相对基准"做快检；本版缺口是自主检出的**动态 bbox**。
因此把"快检(DEFECTIVE)"任务交给检测器（BarrierAlarm 报出即视为缺，否则视为完好），
本模块只负责**时序确认**：候选缺口在"衰减池 + 路露率 RER + 迟滞"下，
把"真移除"与"人/车长期遮挡"分开——二者都表现为带被打断，只有 RER 能分离。

Tuning 来自 tmp/server 实测：非路内容 RER<=0.10（货车 0.00、完好边缘 0.02-0.10），
真移除缺口 RER 0.27-0.65，RER 阈值取 0.25 保留 >=0.15 绝对裕量。
"""
import enum
from typing import List, Optional

import cv2
import numpy as np


class GapCondition(enum.Enum):
    INTACT = 1
    DEFECTIVE = 2


class GapState(enum.Enum):
    NORMAL = 1
    SUSPECTED = 2
    ALARM = 3


class RoadColorProfile:
    """逐帧动态标定的路面 Lab 颜色分布（在 8x8 patch 均值上拟合）。

    用 patch 均值而非像素级：像素级 sigma 夸大纹理噪声、错置 q95 门限，
    在 1333 帧上把真缺口 RER 从 0.47 拉低到 0.27。
    """

    def __init__(self, road_roi_bgr: np.ndarray, patch_size: int = 8) -> None:
        if road_roi_bgr.size == 0:
            raise ValueError("路面色标定框为空。")
        road_lab = cv2.cvtColor(road_roi_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        samples = self._patch_means(road_lab, patch_size)
        self._mean_l = float(np.mean(samples[:, 0]))
        self._std_l = float(np.std(samples[:, 0]))
        self._mean_ab = np.mean(samples[:, 1:], axis=0)
        self._std_ab = np.maximum(np.std(samples[:, 1:], axis=0), 0.1)
        self._l_max = min(255.0, self._mean_l + 2.5 * self._std_l)
        distances = np.sqrt(np.sum(((samples[:, 1:] - self._mean_ab) / self._std_ab) ** 2, axis=1))
        self._dist_threshold = float(np.percentile(distances, 95)) + 0.5

    @staticmethod
    def _patch_means(lab_roi: np.ndarray, patch_size: int) -> np.ndarray:
        h, w, _ = lab_roi.shape
        ny, nx = h // patch_size, w // patch_size
        if ny == 0 or nx == 0:
            return lab_roi.reshape(-1, 3)
        grid = lab_roi[: ny * patch_size, : nx * patch_size]
        return grid.reshape(ny, patch_size, nx, patch_size, 3).swapaxes(1, 2).reshape(
            ny * nx, patch_size, patch_size, 3).mean(axis=(1, 2))

    @property
    def distance_threshold(self) -> float:
        return self._dist_threshold

    def calc_patch_distance(self, patch_mean_lab: np.ndarray) -> float:
        if patch_mean_lab[0] > self._l_max:
            return 999.0
        norm = (patch_mean_lab[1:] - self._mean_ab) / self._std_ab
        return float(np.sqrt(np.sum(norm ** 2)))


def calc_patch_rer(image_bgr, slot_mask, foreground_mask, road_profile,
                   purity=0.70, ps=8) -> float:
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
               if road_profile.calc_patch_distance(m) <= road_profile.distance_threshold)
    return float(road) / float(patch_means.shape[0])


class GapTracker:
    """单个缺口候选的时序状态机：衰减池 + 中值帧 RER 确认 + 迟滞门控 + 快速脱锁。

    阈值全部由调用方注入（见标定文件），此处不硬编码。
    """

    def __init__(self, box: tuple, tcfg: dict) -> None:
        t = tcfg
        self.box = box
        self._decay = t["decay_rate"]
        self._hold = t["alarm_hold_s"]
        self._reconfirm = t["reconfirm_s"]
        self._rer_th = t["rer_threshold"]
        self._reset = t["intact_reset_s"]
        self._cap = t["median_window"]
        self.state = GapState.NORMAL
        self._accum = 0.0
        self._last_update = -1.0
        self._window: List[np.ndarray] = []
        self._rer_cache = (None, 0.0)
        self._rer_below = 0
        self._intact_sec = 0.0
        self.last_seen = 0.0
        self.severity = 0.0
        self.kind = "gap"
        self.row_id = ""

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        self._window.append(frame_bgr)
        if len(self._window) > self._cap:
            self._window.pop(0)

    def median_frame(self) -> Optional[np.ndarray]:
        if not self._window:
            return None
        return np.median(np.stack(self._window, axis=0), axis=0).astype(np.uint8)

    def update(self, condition: GapCondition, now: float, confirm_fn) -> GapState:
        dt = max(0.0, now - self._last_update) if self._last_update >= 0.0 else 0.0
        self._last_update = now

        if condition == GapCondition.DEFECTIVE:
            self._accum += dt
            self._intact_sec = 0.0
        else:
            self._accum = max(0.0, self._accum - dt * self._decay)
            self._intact_sec += dt
            if self._intact_sec >= self._reset:
                self._accum = 0.0
                self._rer_cache = (None, 0.0)
                self._rer_below = 0
                self.state = GapState.NORMAL
                return self.state

        if self._accum <= 0.0:
            self.state = GapState.NORMAL
            self._rer_below = 0
            return self.state

        if self._accum < self._hold:
            self.state = GapState.SUSPECTED
            return self.state

        # 持续到 ALARM_HOLD：对中值帧跑一次 RER 确认（<=RECONFIRM 一次），迟滞 2 次才降级。
        last_t, cached = self._rer_cache
        if last_t is None or now - last_t >= self._reconfirm:
            med = self.median_frame()
            cached = confirm_fn(med) if med is not None else 0.0
            self._rer_cache = (now, cached)
        self._rer_below += 1
        if cached >= self._rer_th:
            self.state = GapState.ALARM
            self._rer_below = 0
        elif self._rer_below >= 2:
            self.state = GapState.SUSPECTED
        return self.state
