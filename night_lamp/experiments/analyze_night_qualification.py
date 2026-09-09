"""P1 Night Qualification evidence analysis (OFFLINE ONLY).

Reads overnight burst evidence, derives candidate T_enter/T_exit + persistence
from data. Does NOT import/touch detector, registry, or any runtime code.
Outputs: figs/fig6_night_qualification.png, p1_night_qualification.json
"""
import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\admin\Desktop\work\BarriGuard\night_lamp\output\overnight_20260908"
FIG = os.path.join(OUT, "figs")

# --- load raw burst evidence (field names verified against actual JSON) ---
files = sorted(glob.glob(os.path.join(OUT, "bursts", "*.json")))
D = []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    if d.get("status") == "OK":
        D.append(d)
assert len(D) == 9, len(D)
# actual collection times (loop-start bids lag ~1h behind collection; snapshots prove cadence)
T = ["18:12", "19:12", "20:12", "21:12", "22:13", "23:13", "00:13", "01:13", "02:13"]
G = np.array([float(d["global_median"]) for d in D])
BIDS = [d["burst_id"] for d in D]
L1306_B = np.array([next(x["profile_B"] for x in d["lamps"] if x["lamp_id"] == "L1306") for d in D])

# --- trajectory ---
dG = np.diff(G)
R = G[1:] / G[:-1]
Gmax, Gmin = G.max(), G.min()
Q = (G - Gmin) / (Gmax - Gmin)
print("| burst | time | global_median | delta | ratio | Q |")
for i in range(len(D)):
    dl = f"{dG[i-1]:+.1f}" if i else "  --"
    ra = f"{R[i-1]:.3f}" if i else " --"
    print(f"| {BIDS[i]} | {T[i]} | {G[i]:.0f} | {dl} | {ra} | {Q[i]:.3f} |")
print(f"initial={Gmax:.0f} final={Gmin:.0f} drop={Gmax-Gmin:.0f} "
      f"monotonic_dec={bool((dG <= 0).all())} max_pos_delta={dG.max():+.1f} "
      f"max_neg_delta={dG.min():+.1f}")

# --- candidates derived from data ---
# B00 G=130 (twilight, L1306 B=1 transient); B01..B08 G in [58..79] (stable night).
# The only observed transition is the unobserved B00->B01 gap (drop 51).
# Candidates therefore sit INSIDE the gap (79,130): midpoint ~104.
cands = {"A": (90.0, 110.0), "B": (100.0, 120.0), "C": (85.0, 105.0)}
print("candidates (T_enter, T_exit):", cands)

# --- persistence replay: TWILIGHT->NIGHT after N consecutive G < T_enter ---
def replay(te, tx, p):
    # state per burst index; exit rule needs G > T_exit while NIGHT
    st, cnt, enter = "TWILIGHT", 0, None
    hist = []
    for i, g in enumerate(G):
        if st == "TWILIGHT":
            cnt = cnt + 1 if g < te else 0
            if cnt >= p:
                st, enter = "NIGHT", i
        else:
            if g > tx:
                st, cnt = "TWILIGHT", 0
        hist.append(st)
    return hist, enter

rows = []
for nm, (te, tx) in cands.items():
    assert te < tx
    for p in (1, 2, 3):
        hist, enter = replay(te, tx, p)
        eb = BIDS[enter] if enter is not None else "NO TRANSITION OBSERVED"
        et = T[enter] if enter is not None else "--"
        rows.append((nm, te, tx, p, eb, et, hist))
        print(f"| {nm} | p={p} | enter={eb} | time={et} |")
    # B00 exclusion check (p=2 shown; full check below)
print("--- B00 exclusion per candidate/persistence ---")
b00 = {}
for nm, te, tx, p, eb, et, hist in rows:
    excl = hist[0] != "NIGHT"
    b00[f"{nm}/p{p}"] = {"B00_L1306": int(L1306_B[0]), "B00_state": hist[0],
                         "excluded": bool(excl)}
    print(f"{nm}/p{p}: B00_state={hist[0]} excluded={excl}")

# --- recommend: gap-midpoint candidate with p=2 (one-burst margin against
# single-dip flicker; still leaves 6 usable night bursts). ---
REC = {"candidate": "B", "T_enter": 100.0, "T_exit": 120.0, "persistence": 2}
_, rent = replay(REC["T_enter"], REC["T_exit"], REC["persistence"])
night_dur_h = (len(D) - 1 - rent) if rent is not None else 0
print(f"recommended: {REC} enter_idx={rent} usable_bursts_after_enter={len(D)-1-rent}")

res = {
    "input_bursts": BIDS,
    "global_series": [{"burst": b, "time": t, "global_median": float(g),
                       "delta": None if i == 0 else float(dG[i-1]),
                       "ratio": None if i == 0 else float(R[i-1]),
                       "Q": float(Q[i])}
                      for i, (b, t, g) in enumerate(zip(BIDS, T, G))],
    "trajectory": {"initial": float(Gmax), "final": float(Gmin), "drop": float(Gmax - Gmin),
                   "monotonic_decreasing": bool((dG <= 0).all()),
                   "max_positive_delta": float(dG.max()), "max_negative_delta": float(dG.min()),
                   "pump_event": False,
                   "pump_evidence": "all deltas <= 0 after B00; largest single drop -51 in "
                                    "unobserved B00->B01 dusk gap; no positive jump > 0"},
    "transition_interval": {"observed": "between B00 (G=130, 18:12) and B01 (G=79, 19:12)",
                            "note": "1-hour sampling cannot resolve minute-level transition; "
                                    "interval only, not a timestamp"},
    "L1306_B": [int(v) for v in L1306_B],
    "threshold_candidates": [{"name": n, "T_enter": te, "T_exit": tx, "hysteresis": tx - te,
                              "basis": b} for (n, (te, tx)), b in
                             zip(cands.items(),
                                 ["lower third of observed gap (79,130)",
                                  "gap midpoint ~104 (recommended family)",
                                  "upper third of gap, closest to dusk point"])],
    "persistence_results": [{"candidate": n, "persistence": p, "enter_burst": eb, "enter_time": et}
                            for n, te, tx, p, eb, et, h in rows],
    "b00_exclusion": b00,
    "recommended_threshold": REC,
    "recommended_persistence": REC["persistence"],
    "b00_excluded": True,
    "night_observed_duration": {"enter_burst": BIDS[rent], "enter_time": T[rent],
                                "last_burst": BIDS[-1],
                                "usable_bursts_after_enter": len(D) - 1 - rent,
                                "approx_hours": round((len(D) - 1 - rent), 1)},
    "limitations": ["9 bursts at ~1h spacing; transition undersampled",
                    "single night, no pump event observed; thresholds provisional",
                    "B00 timestamp is loop-start; collection ~40 frames later (negligible vs 1h)",
                    "fps field B01+ invalid (separate code bug); does not affect G series"],
}
json.dump(res, open(os.path.join(OUT, "p1_night_qualification.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# --- fig6 ---
X = np.arange(len(D))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(X, G, "o-", color="k", label="global_median")
ax.axhline(REC["T_enter"], color="C2", ls="--", label=f"T_enter={REC['T_enter']}")
ax.axhline(REC["T_exit"], color="C3", ls="--", label=f"T_exit={REC['T_exit']}")
ax.axvspan(-0.5, 0.5, color="C1", alpha=0.15, label="TWILIGHT (B00)")
ax.axvspan(0.5, len(D) - 0.5, color="C2", alpha=0.08, label="NIGHT (B01+)")
ax.annotate("B00 L1306 B=1\nexcluded by TWILIGHT", xy=(0, G[0]), xytext=(2.2, G[0] + 8),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate(f"NIGHT enter {T[rent]} (p={REC['persistence']})", xy=(rent, REC["T_enter"]),
            xytext=(rent + 0.5, REC["T_enter"] + 18), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.set_xticks(X); ax.set_xticklabels(T, rotation=30)
ax.set_ylabel("global_median"); ax.set_title("P1: night qualification evidence (provisional)")
ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_night_qualification.png"), dpi=120)
print("wrote fig6 + p1_night_qualification.json")
