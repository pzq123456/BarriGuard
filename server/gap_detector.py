"""Slot 缺口检测：基准差分快检 + 中值帧 RER 确认 + 迟滞状态机。

由 lab/gap_detector.py 提升入生产（离线 6 场景回归 6/6 + 实流 90s x3 验证）。
标定依据：非路内容 RER<=0.10（货车 0.00）vs 真缺口 0.27-0.65，帧 0944/1333。
"""
import enum
from typing import List, Optional

import cv2
import numpy as np


class SlotCondition(enum.Enum):
    INTACT = 1
    DEFECTIVE = 2


class SlotState(enum.Enum):
    NORMAL = 1
    SUSPECTED = 2
    ALARM = 3


# Core Constants
CONST_MIN_PATCH_PURITY: float = 0.70
CONST_DEFAULT_PATCH_SIZE: int = 8
CONST_DECAY_RATE_PER_SEC: float = 0.5
CONST_ALARM_HOLD_SEC: float = 10.0
CONST_RELATIVE_COVER_RATIO: float = 0.50
CONST_MIN_MISSING_AREA_PX: int = 200
# Measured separation (frames 0944/1333, 5 calibrations): non-road content RER
# <= 0.10 (truck 0.00, intact-slot edges 0.02-0.10) vs real gap 0.27-0.65.
# 0.25 keeps >=0.15 absolute margin to non-road; revalidate across day times.
CONST_DEFAULT_RER_THRESHOLD: float = 0.25
# Suspicion older than this alarms regardless of RER: appearance cannot separate
# a long-parked vehicle from a removal; escalation bounds the ambiguity, operator
# reviews. Re-confirm RER at most this often (median-frame RER is expensive).
CONST_ESCALATE_SEC: float = 300.0
CONST_RECONFIRM_SEC: float = 5.0


class RoadColorProfile:
    """Frame-wise dynamically calibrated road color distribution in Lab space.

    Fits on 8x8 patch means (not raw pixels): pixel-level sigma overstates
    texture noise and misplaces the q95 gate, which collapsed real-gap RER
    from 0.47 to 0.27 on frame 1333.
    """

    def __init__(self, road_roi_bgr: np.ndarray,
                 patch_size: int = CONST_DEFAULT_PATCH_SIZE) -> None:
        """Fits mean/std and distance threshold dynamically from frame's road ROI."""
        if road_roi_bgr.size == 0:
            raise ValueError("Road ROI provided for dynamic color calibration is empty.")

        road_lab = cv2.cvtColor(road_roi_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        samples = self._patch_means(road_lab, patch_size)

        self._mean_l: float = float(np.mean(samples[:, 0]))
        self._std_l: float = float(np.std(samples[:, 0]))
        self._mean_ab: np.ndarray = np.mean(samples[:, 1:], axis=0)
        self._std_ab: np.ndarray = np.maximum(np.std(samples[:, 1:], axis=0), 0.1)

        # L gate upper bound only: rejects white/bright occluders (trucks). The lower
        # bound is left to the a/b distance gate so shadowed road is not killed.
        self._l_max: float = min(255.0, self._mean_l + 2.5 * self._std_l)

        # Distance threshold from q95 + 0.5 margin of the road patches
        distances = np.sqrt(np.sum(((samples[:, 1:] - self._mean_ab) / self._std_ab) ** 2, axis=1))
        q95 = float(np.percentile(distances, 95))
        self._dist_threshold: float = q95 + 0.5

    @staticmethod
    def _patch_means(lab_roi: np.ndarray, patch_size: int) -> np.ndarray:
        """8x8 patch means; falls back to raw pixels when ROI is too small."""
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
        """Calculates Mahalanobis distance with dynamic thresholding."""
        patch_l = patch_mean_lab[0]
        if patch_l > self._l_max:
            return 999.0

        patch_ab = patch_mean_lab[1:]
        normalized_diff = (patch_ab - self._mean_ab) / self._std_ab
        return float(np.sqrt(np.sum(normalized_diff**2)))


class SlotTracker:
    """Maintains state with hysteresis decay and windowed median frame buffering."""

    def __init__(self, slot_id: int, base_coverage: float, window_capacity: int = 5) -> None:
        self._slot_id: int = slot_id
        self._base_coverage: float = max(base_coverage, 0.05)
        self._state: SlotState = SlotState.NORMAL
        self._accumulated_abnormal_sec: float = 0.0
        self._last_update_time: float = -1.0
        self._frame_window: List[np.ndarray] = []
        self._window_capacity: int = window_capacity
        self._rer_cache: tuple = (None, 0.0)

    @property
    def state(self) -> SlotState:
        return self._state

    @property
    def base_coverage(self) -> float:
        return self._base_coverage

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Pushes current frame into window buffer for temporal median calculation."""
        self._frame_window.append(frame_bgr)
        if len(self._frame_window) > self._window_capacity:
            self._frame_window.pop(0)

    def get_temporal_median_frame(self) -> Optional[np.ndarray]:
        """Calculates pixel-wise temporal median frame across buffered window."""
        if not self._frame_window:
            return None
        stacked = np.stack(self._frame_window, axis=0)
        return np.median(stacked, axis=0).astype(np.uint8)

    def update(
        self,
        condition: SlotCondition,
        current_time: float,
        rer_evaluator_fn,
        alarm_hold_sec: float = CONST_ALARM_HOLD_SEC,
        rer_threshold: float = CONST_DEFAULT_RER_THRESHOLD,
    ) -> SlotState:
        """Updates state machine: accumulator gate + RER fast gate + escalation backstop."""
        dt = max(0.0, current_time - self._last_update_time) if self._last_update_time >= 0.0 else 0.0
        self._last_update_time = current_time

        if condition == SlotCondition.DEFECTIVE:
            self._accumulated_abnormal_sec += dt
        else:
            self._accumulated_abnormal_sec = max(
                0.0, self._accumulated_abnormal_sec - dt * CONST_DECAY_RATE_PER_SEC
            )
            self._rer_cache = (None, 0.0)

        acc = self._accumulated_abnormal_sec
        if acc <= 0.0:
            self._state = SlotState.NORMAL
            return self._state

        if acc >= CONST_ESCALATE_SEC:
            self._state = SlotState.ALARM
            return self._state

        if acc >= alarm_hold_sec:
            # RER confirms onset only; a latched ALARM is never re-gated, else a
            # dusk-light RER dip would downgrade a confirmed incident mid-flight.
            if self._state is SlotState.ALARM:
                return self._state
            last_t, cached_rer = self._rer_cache
            if last_t is None or current_time - last_t >= CONST_RECONFIRM_SEC:
                median_frame = self.get_temporal_median_frame()
                cached_rer = rer_evaluator_fn(median_frame) if median_frame is not None else 0.0
                self._rer_cache = (current_time, cached_rer)
            self._state = SlotState.ALARM if cached_rer >= rer_threshold else SlotState.SUSPECTED
            return self._state

        # ALARM latches until the accumulator fully decays: a transient covering a
        # confirmed gap must not downgrade an ongoing incident to SUSPECTED.
        if self._state is SlotState.ALARM:
            return self._state

        self._state = SlotState.SUSPECTED
        return self._state


def calc_patch_rer_vectorized(
    image_bgr: np.ndarray,
    slot_mask: np.ndarray,
    foreground_mask: np.ndarray,
    road_profile: RoadColorProfile,
    patch_size: int = CONST_DEFAULT_PATCH_SIZE,
) -> float:
    """Calculates Road Exposure Ratio using corrected vectorized array dimensions."""
    missing_mask = cv2.bitwise_and(slot_mask, cv2.bitwise_not(foreground_mask))
    missing_area_px = cv2.countNonZero(missing_mask)

    if missing_area_px < CONST_MIN_MISSING_AREA_PX or missing_area_px < (patch_size * patch_size):
        return 0.0

    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    h, w, _ = image_lab.shape
    num_y = h // patch_size
    num_x = w // patch_size

    cropped_lab = image_lab[: num_y * patch_size, : num_x * patch_size]
    cropped_mask = missing_mask[: num_y * patch_size, : num_x * patch_size]

    lab_patches = cropped_lab.reshape(num_y, patch_size, num_x, patch_size, 3).swapaxes(1, 2)
    mask_patches = cropped_mask.reshape(num_y, patch_size, num_x, patch_size).swapaxes(1, 2)

    valid_patch_counts = np.count_nonzero(mask_patches, axis=(2, 3))
    purity_threshold = CONST_MIN_PATCH_PURITY * patch_size * patch_size
    valid_patch_mask = valid_patch_counts >= purity_threshold

    if not np.any(valid_patch_mask):
        return 0.0

    # FIX P0: collapse both patch spatial dims; previous axis=1 left shape [N, 8, 3] and crashed
    selected_patches = lab_patches[valid_patch_mask]
    patch_means = np.mean(selected_patches, axis=(1, 2))

    road_count = 0
    total_valid = patch_means.shape[0]

    for mean_lab in patch_means:
        dist = road_profile.calc_patch_distance(mean_lab)
        if dist <= road_profile.distance_threshold:
            road_count += 1

    return float(road_count) / float(total_valid) if total_valid > 0 else 0.0


def evaluate_slot_fast(
    slot_mask: np.ndarray,
    foreground_mask: np.ndarray,
    base_coverage: float,
) -> SlotCondition:
    """Evaluates slot coverage fast against per-slot relative threshold."""
    slot_area = cv2.countNonZero(slot_mask)
    if slot_area == 0:
        return SlotCondition.INTACT

    covered = cv2.countNonZero(cv2.bitwise_and(slot_mask, foreground_mask))
    current_coverage = float(covered) / float(slot_area)

    threshold = base_coverage * CONST_RELATIVE_COVER_RATIO
    if current_coverage < threshold:
        return SlotCondition.DEFECTIVE

    return SlotCondition.INTACT
