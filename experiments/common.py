"""实验公共工具：目录约定、图像加载、run 目录、裁剪。

目录约定：
  data/input/  原始帧，命名 <相机>_<时分>.png，只进不改
  results/     所有试验输出，每轮一个 <时间戳>_<实验名>/ 子目录，互不覆盖
"""
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
RESULTS_DIR = ROOT / "results"


def load_input(name: str) -> np.ndarray:
    img = cv.imread(str(INPUT_DIR / name))
    if img is None:
        raise FileNotFoundError(f"{INPUT_DIR / name} 不存在")
    return img


def new_run_dir(exp: str) -> Path:
    run = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{exp}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def save(img: np.ndarray, run: Path, name: str) -> Path:
    path = run / name
    cv.imwrite(str(path), img)
    return path


def crop_pad(img: np.ndarray, box: tuple, pad: int = 16) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    return img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
