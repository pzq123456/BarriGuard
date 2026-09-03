"""展示渲染：mask 绿色叠加 + slot 状态框。"""
import cv2 as cv
import numpy as np

from .gap_detector import SlotState

TINT, ALPHA = (80, 220, 80), 0.45
STATE_COLOR = {SlotState.NORMAL: (120, 200, 120), SlotState.SUSPECTED: (80, 160, 255),
               SlotState.ALARM: (80, 80, 255)}
LABEL_FONT = cv.FONT_HERSHEY_SIMPLEX


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    layer = frame.copy()
    layer[mask > 0] = TINT
    return cv.addWeighted(layer, ALPHA, frame, 1 - ALPHA, 0)


def draw_slots(vis: np.ndarray, slot_states: dict) -> np.ndarray:
    """slot_states: {name: (rect, SlotState)}，框色随状态、标签用 ASCII。"""
    for name, (rect, state) in slot_states.items():
        x0, y0, x1, y1 = rect
        color = STATE_COLOR[state]
        cv.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv.putText(vis, f"{name}:{state.name}", (x0, y0 - 6),
                   LABEL_FONT, 0.6, color, 2)
    return vis
