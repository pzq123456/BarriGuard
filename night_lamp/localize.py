import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).parent
r = json.load(open(HERE / "pipeline_result.json"))
far = [a for a in r["alive_flash"] if 1150 <= a["x"] <= 1900 and 360 <= a["y"] <= 470]
far.sort(key=lambda a: a["x"])
print("far-line alive lamps (x, y, duty, lag, ac, dx_to_prev):")
prev = None
for a in far:
    d = f"{a['x'] - prev:5.1f}" if prev else "    -"
    print(f"  ({a['x']:7.1f},{a['y']:6.1f}) duty={a['duty']*100:5.1f}% lag={a['lag']} "
          f"ac={a['ac']}  dx={d}")
    prev = a["x"]
clean = [a for a in far if 1220 <= a["x"] <= 1560]
sp = np.diff([a["x"] for a in clean])
print("clean section x1220-1560 spacings:", np.round(sp, 1), "median", np.median(sp))
