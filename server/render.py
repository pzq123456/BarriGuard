"""通用渲染：算法无关，只认 server.algo.Annotation。

level 通用配色：alarm 红 / suspected 橙 / info 绿。
overlay 是任意二值掩膜的绿色叠加（算法可通过 debug 层提供 roi 掩膜）。
"""
import cv2 as cv
import numpy as np

TINT, ALPHA = (80, 220, 80), 0.45
LEVEL_COLOR = {"alarm": (80, 80, 255), "suspected": (80, 160, 255),
               "info": (120, 200, 120)}


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    layer = frame.copy()
    layer[mask > 0] = TINT
    return cv.addWeighted(layer, ALPHA, frame, 1 - ALPHA, 0)


def draw_annots(vis: np.ndarray, annots) -> np.ndarray:
    """画算法标注：box 画框，point 画圈（给以后 night_lamp 灯点位预留）。"""
    for a in annots:
        color = LEVEL_COLOR.get(a.level, (200, 200, 200))
        if a.kind == "box" and len(a.box) == 4:
            x0, y0, x1, y1 = a.box
            cv.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv.putText(vis, a.label, (x0, max(20, y0 - 6)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif a.kind == "point" and len(a.box) == 2:
            x, y = a.box
            cv.circle(vis, (int(x), int(y)), 7, color, 2)
            cv.putText(vis, a.label, (int(x) + 9, int(y) - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def draw_status(vis: np.ndarray, status: str) -> np.ndarray:
    """非 OK 状态（SUNGLARE 等）在画面压横幅：盲区必须看得见，不能静默。"""
    if status == "OK":
        return vis
    cv.putText(vis, status, (20, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    return vis
