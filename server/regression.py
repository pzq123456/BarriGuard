"""部署验收回归：1750 已知无缺口零 alarm；1749 固定缺口必 alarm。

跑法：同帧×14 帧（dt=1s，加速时序收敛）。注意这是静态图验收，只能验证
“稳态 shadow-FP 静默 + 固定缺口报警”；框内有人/车等 transient 遮挡在静态图上
与真缺口不可分（靠线上时间衰减消），若某帧仅因此 FAIL，需看图确认。

用法：python -m server.regression [--config PATH]，exit 0 全过。
"""
import argparse
import glob
import os
import sys

import cv2 as cv

from .config import load
from .registry import create

REPEAT = 14

EXPECT = [
    ("1750", "data/1750/*.jpg", 0),   # 已知无缺口：零 alarm
    ("1749", "data/1749/*.jpg", 1),   # 固定缺口：至少一 alarm
]

# 静态图按设计分不清 transient 遮挡与真缺口（靠线上时间衰减消），这类帧除外，
# 由人工看框图确认。治理规则（防回归集烂掉）：
# 1. 每条必须写 reason（为什么静态无解）+ covered_by（线上靠哪层兜底）；
# 2. 超过 EXCLUDE_CAP 条直接 FAIL，触发专项复盘，不得无限累加。
EXCLUDE_CAP = 5
EXCLUDE = {
    # 09:47 白色货车停在左排后遮挡水马（框 678,120,719,188），静态必报，车走即消。
    "dahua1002491_20260910_094708_01.jpg": {
        "reason": "白色货车遮挡",
        "covered_by": "时序衰减（intact_reset/track_stale，车走即消；02/03 同分钟帧已无报警）",
    },
}


def check(params):
    ok_all, lines = True, []
    if len(EXCLUDE) > EXCLUDE_CAP:
        return False, [("FAIL", "-", f"EXCLUDE {len(EXCLUDE)} 条超上限 {EXCLUDE_CAP}，触发复盘", "", "")]
    for cid, pat, want in EXPECT:
        calib = next(c for c in params["cameras"] if c["id"] == cid)["algos"]["water_gap"]
        for img in sorted(glob.glob(pat)):
            base = os.path.basename(img)
            if base in EXCLUDE:
                ex = EXCLUDE[base]
                if not ex.get("reason") or not ex.get("covered_by"):
                    return False, [("FAIL", cid, base, "EXCLUDE 缺 reason/covered_by", "")]
                lines.append(("SKIP", cid, base, ex["reason"], ex["covered_by"]))
                continue
            frame = cv.imread(img)
            algo = create("water_gap", frame.shape, calib, cid)
            for i in range(REPEAT):
                res = algo.step(frame, float(i))
            alarms = [e for e in res.events if e.kind == "alarm"]
            good = (len(alarms) == 0) if want == 0 else (len(alarms) >= 1)
            ok_all = ok_all and good
            lines.append(("PASS" if good else "FAIL", cid, os.path.basename(img),
                          [(e.payload.get("row_id"), e.payload.get("rer")) for e in alarms],
                          res.debug.get("frame_status")))
    tok, tlines = check_temporal(params)
    return ok_all and tok, lines + tlines


def check_temporal(params):
    """时序行为钉死（R5）：同一货车停留序列（30s），hold=10 必漏报一次、
    hold=60 必全程无声。钉的是累积/衰减速率，静态帧回归测不出来。
    hold=10 漏报是已知机制行为（transient 熬过短门限），不是回归失败；
    若未来 decay/hold 语义有意改变，需同步更新此处期望并写明原因。
    """
    calib = next(c for c in params["cameras"] if c["id"] == "1750")["algos"]["water_gap"]
    seq = (["data/1750/clean_01.jpg"] * 2
           + ["data/1750/dahua1002491_20260910_094708_01.jpg"] * 3
           + ["data/1750/dahua1002491_20260910_094708_02.jpg",
              "data/1750/dahua1002491_20260910_094708_03.jpg"]
           + ["data/1750/clean_01.jpg"] * 3)
    frames = [cv.imread(p) for p in seq]
    ok_all, lines = True, []
    for hold, want_alarm in ((10.0, True), (60.0, False)):
        cal = {**calib, "track": {**calib["track"], "alarm_hold_s": hold}}
        algo = create("water_gap", frames[0].shape, cal, "1750")
        saw = False
        for i, fr in enumerate(frames):
            res = algo.step(fr, float(i * 10))
            saw = saw or any(e.kind == "alarm" for e in res.events)
        good = saw == want_alarm
        ok_all = ok_all and good
        lines.append(("PASS" if good else "FAIL", "1750", f"van-series hold={hold:g}",
                      f"alarm={saw} expect={want_alarm}", ""))
    return ok_all, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    ok, lines = check(load(args.config))
    for st, cid, img, alarms, status in lines:
        print(st, cid, img, alarms, status)
    print("ALL-PASS" if ok else "HAS-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
