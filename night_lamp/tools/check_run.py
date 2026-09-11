"""Inspect a production run -- reads bursts/*.json + night_state.json, prints rates.

No matplotlib, no detector import. Evidence reader only.

  python tools/check_run.py --out output/production_1749
"""
import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lamps", nargs="*", default=["M06", "M02"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.out, "bursts", "*.json")))
    print("bursts: %s" % len(files))
    g = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if d.get("status") != "OK":
            print("  %s status=%s" % (d.get("burst_id"), d.get("status")))
            continue
        g.append(d["global_median"])
        ns = d.get("night_state", {})
        print("  %s G=%s night=%s A/B: %s" %
              (d["burst_id"], d["global_median"], ns.get("state"),
               [(l["lamp_id"], l["profile_A"], l["profile_B"])
                for l in d["lamps"] if l["lamp_id"] in args.lamps]))
    ns_path = os.path.join(args.out, "night_state.json")
    if os.path.isfile(ns_path):
        print("persisted:", json.load(open(ns_path, encoding="utf-8")))
    s = os.path.join(args.out, "summary.json")
    if os.path.isfile(s):
        d = json.load(open(s, encoding="utf-8"))
        print("summary: total_bursts=%s" % d["total_bursts"])
        for lid, p in d.get("controls", d.get("P0", {})).items():
            print("  %s A=%.3f B=%.3f n_on=%.1f" %
                  (lid, p["A_rate"], p["B_rate"], p["mean_n_on"]))


if __name__ == "__main__":
    main()
