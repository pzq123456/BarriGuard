"""Build registry CSV from manual LabelMe boxes -- truthful, no invention.

v2 schema (every column is used, see header comment in output CSV):
  kind       lamp|watchlist. Loader splits: lamp->detector, watchlist->candidates.
  id         Stable key. Lamps M01..M29 sorted by x (sequence, NOT coordinate).
             Watchlist keeps legacy Wxxxx. Used in logs/overlay/summary.
  x,y        Box center, 1 decimal. Detector ROI center.
  w,h        Box size, 1 decimal (lamp only). Audit: proves what was annotated.
             Propagated to lamp_metrics so it is not a dead column.
  origin     Machine-readable provenance. manual:<file>#idxNN or
             legacy:<file>#<old_id>. Used by --check for 1:1 audit and by
             future merges to avoid duplicates. Replaces prose provenance.
  status     active|voided. Loader detects only active. Voided rows are
             tombstones: kept for audit, never deleted. Future route for
             algorithm/human review without losing truth.
  updated_at YYYY-MM-DD of last row edit. Audit / staleness.
  note       Factual audit only, no "?". Lamps: box + neighbor gaps + REVIEW
             flag if anomalous. Propagated to output so it is read, not dead.

Dropped v1 columns and why:
  r          Constant 10, detector uses config detector.roi_r. Per-lamp r lied.
  type/band  Research guesses (steady/flash, near/far), never fed detector.
             Behavior belongs to future algorithm output, not manual truth.
  role       Mixed table-name ("registry") with controls (P0/P1). Controls
             move to config registry.controls. Evidence colors by controls.
  provenance Free prose, same string on all rows. Replaced by origin.

Usage:
  python tools/build_registry.py --json ../../data/test/night_longexp_short.json --out ../configs/registry_1749.csv
  python tools/build_registry.py --check --json ../../data/test/night_longexp_short.json --out ../configs/registry_1749.csv
  python tools/build_registry.py --mark-voided M07 --reason "field check ..." --out ../configs/registry_1749.csv
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HEADER = ["kind", "id", "x", "y", "w", "h", "origin", "status", "updated_at", "note"]

# REVIEW flags audit only, never change status. Perspective makes near gaps
# large, so only flag far-line breaks where both ends are x>=1200.
FAR_X = 1200.0
GAP_WARN_PX = 50.0


def load_boxes(json_path):
    """Return [(idx, x0, y0, x1, y1)] for label==light rectangles only."""
    d = json.load(open(json_path, encoding="utf-8"))
    out = []
    for i, s in enumerate(d.get("shapes", [])):
        if s.get("label") != "light":
            continue
        if s.get("shape_type") != "rectangle":
            raise SystemExit("shape %d not rectangle: %s" % (i, s.get("shape_type")))
        pts = s["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((i, min(xs), min(ys), max(xs), max(ys)))
    if not out:
        raise SystemExit("no light boxes in %s" % json_path)
    return out


def build_lamp_rows(boxes, json_name, today):
    """Sort by x, assign M01.., compute center/size, gap audit notes."""
    boxes = sorted(boxes, key=lambda t: (t[1] + t[3]) / 2)
    rows = []
    centers = [((x0 + x1) / 2, (y0 + y1) / 2) for _, x0, y0, x1, y1 in boxes]
    for seq, (idx, x0, y0, x1, y1) in enumerate(boxes, 1):
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        mid = "%s#idx%02d" % (json_name, idx)
        note_bits = ["box %.1fx%.1f" % (w, h)]
        warn = []
        if seq > 1:
            dx = cx - centers[seq - 2][0]
            note_bits.append("gap_prev %.1fpx" % dx)
            if cx >= FAR_X and centers[seq - 2][0] >= FAR_X and dx > GAP_WARN_PX:
                warn.append("REVIEW far-gap %.0fpx>%.0f" % (dx, GAP_WARN_PX))
        if seq < len(boxes):
            dx = centers[seq][0] - cx
            note_bits.append("gap_next %.1fpx" % dx)
            if cx >= FAR_X and centers[seq][0] >= FAR_X and dx > GAP_WARN_PX:
                warn.append("REVIEW far-gap %.0fpx>%.0f" % (dx, GAP_WARN_PX))
        if warn:
            note_bits.append("; ".join(sorted(set(warn))))
        rows.append({
            "kind": "lamp",
            "id": "M%02d" % seq,
            "x": "%.1f" % cx,
            "y": "%.1f" % cy,
            "w": "%.1f" % w,
            "h": "%.1f" % h,
            "origin": "manual:%s" % mid,
            "status": "active",
            "updated_at": today,
            "note": "; ".join(note_bits),
        })
    return rows


def carry_watchlist(old_csv, today):
    """Preserve legacy watchlist rows, clean notes, map to v2 columns."""
    if not old_csv or not os.path.isfile(old_csv):
        return []
    out = []
    with open(old_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("kind") or "").strip() != "watchlist":
                continue
            note = (r.get("note") or "").strip().rstrip("?").strip()
            # "?" -> explicit pending-review wording, no punctuation guessing.
            if (r.get("note") or "").strip().endswith("?"):
                note += "; needs field review"
            if not note:
                note = "legacy watchlist; needs field review"
            out.append({
                "kind": "watchlist",
                "id": r["id"].strip(),
                "x": "%.1f" % float(r["x"]),
                "y": "%.1f" % float(r["y"]),
                "w": "",
                "h": "",
                "origin": "legacy:registry_1749.csv#%s" % r["id"].strip(),
                "status": "active",
                "updated_at": today,
                "note": note,
            })
    return out


def write_csv(path, lamp_rows, wl_rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in lamp_rows + wl_rows:
            w.writerow(r)


def cmd_build(args):
    boxes = load_boxes(args.json)
    today = datetime.date.today().isoformat()
    json_name = os.path.basename(args.json)
    lamps = build_lamp_rows(boxes, json_name, today)
    # Backup old file before overwrite (destructive guard).
    if os.path.isfile(args.out):
        bak = args.out + ".bak-%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(args.out, bak)
        print("backup %s" % bak)
        wl = carry_watchlist(args.out, today)
    else:
        wl = carry_watchlist(args.carry_watchlist, today) if args.carry_watchlist else []
    write_csv(args.out, lamps, wl)
    print("wrote %d lamp + %d watchlist -> %s" % (len(lamps), len(wl), args.out))
    for r in lamps:
        if "REVIEW" in r["note"]:
            print("  %s (%s,%s) %s [%s]" % (r["id"], r["x"], r["y"], r["note"], r["origin"]))


def cmd_check(args):
    boxes = load_boxes(args.json)
    json_name = os.path.basename(args.json)
    want = set("manual:%s#idx%02d" % (json_name, i) for i, *_ in boxes)
    with open(args.out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and list(rows[0].keys()) == HEADER, "header mismatch: %s" % list(rows[0].keys())
    lamps = [r for r in rows if r["kind"] == "lamp"]
    got = set(r["origin"] for r in lamps)
    ids = [r["id"] for r in lamps]
    assert len(ids) == len(set(ids)), "dup id"
    assert sorted(ids) == ids, "ids not sorted (must be M01.. by x)"
    missing = want - got
    extra = set(r["origin"] for r in lamps if r["origin"].startswith("manual:")) - want
    print("json boxes=%d csv lamps=%d watchlist=%d" % (len(want), len(lamps), len(rows) - len(lamps)))
    if missing:
        print("MISSING origins: %s" % sorted(missing))
    if extra:
        print("EXTRA origins: %s" % sorted(extra))
    # Center fidelity: each manual row must match its json box center <0.15px.
    by_origin = {r["origin"]: r for r in lamps}
    bad = 0
    for idx, x0, y0, x1, y1 in boxes:
        o = "manual:%s#idx%02d" % (json_name, idx)
        r = by_origin.get(o)
        if not r:
            bad += 1
            continue
        dx = abs(float(r["x"]) - (x0 + x1) / 2)
        dy = abs(float(r["y"]) - (y0 + y1) / 2)
        if max(dx, dy) > 0.15:
            print("DRIFT %s dx=%.2f dy=%.2f" % (o, dx, dy))
            bad += 1
    if missing or extra or bad:
        raise SystemExit("check FAILED")
    print("check OK: 1:1 faithful, centers <0.15px")


def cmd_void(args):
    with open(args.out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == HEADER, "not v2 csv"
    today = datetime.date.today().isoformat()
    hit = [r for r in rows if r["id"] == args.mark_voided]
    if not hit:
        raise SystemExit("id not found: %s" % args.mark_voided)
    r = hit[0]
    r["status"] = "voided" if args.void else "active"
    r["updated_at"] = today
    r["note"] = (r["note"] + "; %s %s: %s" % (
        "voided" if args.void else "reactivated", today, args.reason)).strip("; ")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    print("%s %s: %s" % (r["id"], r["status"], args.reason))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../../data/test/night_longexp_short.json")
    ap.add_argument("--out", default="../configs/registry_1749.csv")
    ap.add_argument("--carry-watchlist", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mark-voided", default=None)
    ap.add_argument("--void", action="store_true", default=True)
    ap.add_argument("--no-void", dest="void", action="store_false")
    ap.add_argument("--reason", default="")
    args = ap.parse_args()
    # Resolve relative to this file (night_lamp/tools/).
    here = os.path.dirname(os.path.abspath(__file__))
    for k in ("json", "out", "carry_watchlist"):
        v = getattr(args, k)
        if v and not os.path.isabs(v):
            setattr(args, k, os.path.normpath(os.path.join(here, v)))
    if args.check:
        cmd_check(args)
    elif args.mark_voided:
        if not args.reason:
            raise SystemExit("--reason required")
        cmd_void(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
