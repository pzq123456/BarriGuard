"""Offline validation of the realtime alarm path.

Feeds cap_day frames through BarrierAlarm.step() exactly as the server
framework would, then dumps the support summary + alarm overlay.
Outputs stay inside water_barrier/output/.

Usage:
    python -m water_barrier.main
    python -m water_barrier.main --limit 20 --out-dir water_barrier/output
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .alarm import REPORT_ROWS, BarrierAlarm
from .roi import ROW_IDS

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cap-dir", default=str(ROOT / "tmp/cap_day"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "output"))
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    seq = sorted(Path(args.cap_dir).glob("cap_*.png"))
    if args.limit > 0:
        seq = seq[:args.limit]

    alarm = BarrierAlarm()
    transitions, prev = 0, None
    last = None
    for p in seq:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        last = bgr
        cur = [(g.row_id, g.sta) for g in alarm.step(bgr)]
        if prev is not None and cur != prev:
            transitions += 1
        prev = cur

    sup = alarm.support
    res = {"n_obs": alarm.n_obs, "transitions": transitions, "rows": {}}
    for rid in ROW_IDS:
        res["rows"][rid] = {
            "boxes": [list(g.sta) for g in alarm._alarms if g.row_id == rid],
            "support_max": round(float(sup[rid].max()), 3),
            "support_mean": round(float(sup[rid].mean()), 4),
            "report": rid in REPORT_ROWS,
        }

    out = Path(args.out_dir)
    fig = out / "ROUND6"
    fig.mkdir(parents=True, exist_ok=True)
    np.save(str(out / "GAPWATCH_support.npy"), np.stack([sup[r] for r in ROW_IDS]))

    overlay = last.copy()
    for g in alarm._alarms:
        x0, y0, x1, y1 = g.box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
        cv2.putText(overlay, f"{g.row_id} {g.support}", (x0, max(20, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(fig / "gap_boxes_mid.jpg"), overlay)

    res["elapsed_s"] = round(time.time() - t0, 1)
    (out / "GAPWATCH_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print(json.dumps(res["rows"], indent=1))


if __name__ == "__main__":
    main()
