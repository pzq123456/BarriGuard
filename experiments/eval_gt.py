"""独立 GT 像素级评测：Precision / Recall / IoU / FPR。

用途：打破自证循环（疑点6）。GT 放 data/gt/<帧名>_gt.png，多边形源文件
data/gt/gt_polys.py，由 make_gt.py 渲染。
用法：python experiments/eval_gt.py <pred.png> <gt.png>
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def evaluate_mask(pred, gt):
    pred_bin, gt_bin = pred > 0, gt > 0
    tp = np.count_nonzero(pred_bin & gt_bin)
    fp = np.count_nonzero(pred_bin & ~gt_bin)
    fn = np.count_nonzero(~pred_bin & gt_bin)
    tn = np.count_nonzero(~pred_bin & ~gt_bin)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": tp / max(1, tp + fp),
            "Recall": tp / max(1, tp + fn),
            "IoU": tp / max(1, tp + fp + fn),
            "FPRate": fp / max(1, fp + tn)}


def main():
    if len(sys.argv) != 3:
        sys.exit("用法: python experiments/eval_gt.py <pred.png> <gt.png>")
    pred = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    gt = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)
    if pred is None or gt is None:
        sys.exit("文件读取失败")
    if pred.shape != gt.shape:
        sys.exit(f"尺寸不一致: pred {pred.shape} vs gt {gt.shape}")

    m = evaluate_mask(pred, gt)
    print(f"Precision {m['Precision']:.4f}   Recall {m['Recall']:.4f}   IoU {m['IoU']:.4f}")
    print(f"FP {m['FP']} px (rate {m['FPRate']:.6f})   FN {m['FN']} px")


if __name__ == "__main__":
    main()
