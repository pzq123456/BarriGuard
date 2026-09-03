"""展示渲染：mask 绿色叠加。"""
import cv2 as cv
import numpy as np

TINT, ALPHA = (80, 220, 80), 0.45


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    layer = frame.copy()
    layer[mask > 0] = TINT
    return cv.addWeighted(layer, ALPHA, frame, 1 - ALPHA, 0)
