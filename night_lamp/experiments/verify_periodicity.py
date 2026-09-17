"""Independent verification of night_lamp.periodicity (hardcoded, no CLI).

Checks, with no edits to the sources under test:
  A. ac_limited(on,5,12) reproduces detector.ac_score(on,5,12) exactly
     (rounded peak + lag) on >=500 random binary series.
  B. clean_train verdicts on explicit synthetic AC inputs built by calling
     ac_limited on binary series: periodic -> True, one-off / irregular /
     high-duty slow -> False.
  C. onsets_of counts 0->1 transitions (plus a leading 1).

Prints PASS/FAIL per group and exits 1 on any FAIL.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from night_lamp import detector  # noqa: E402
from night_lamp.periodicity import (  # noqa: E402
    ac_limited,
    clean_train,
    onsets_of,
)

LAG_LO, LAG_HI = 5, 12
FAILS = []


def group(name, ok):
    print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if not ok:
        FAILS.append(name)


def ac_curve(on, lag_lo, lag_hi):
    peak, lag, curve = ac_limited(on, lag_lo, lag_hi)
    return peak, lag, curve


def verify_equivalence(n_trials=600):
    rng = np.random.default_rng(20260908)
    max_dpk, lag_bad, checked, degenerate = 0.0, 0, 0, 0
    for _ in range(n_trials):
        n = int(rng.integers(20, 601))
        r = rng.random()
        if r < 0.05:
            on = np.zeros(n, np.uint8)
        elif r < 0.10:
            on = np.ones(n, np.uint8)
        else:
            p = float(rng.uniform(0.02, 0.98))
            on = (rng.random(n) < p).astype(np.uint8)

        peak, lag, _ = ac_limited(on, LAG_LO, LAG_HI)
        ref_peak, ref_lag = detector.ac_score(on, LAG_LO, LAG_HI)

        x = on.astype(float) - on.mean()
        if float(x @ x) <= 0:
            degenerate += 1
            continue

        checked += 1
        dpk = abs(round(peak, 3) - ref_peak)
        max_dpk = max(max_dpk, dpk)
        if round(peak, 3) != ref_peak or lag != ref_lag:
            lag_bad += 1
            print("  MISMATCH n=%d: ac_limited=(%.6f,%d) "
                  "detector=(%.3f,%d)"
                  % (n, peak, lag, ref_peak, ref_lag))

    print("  trials=%d  non-degenerate=%d  degenerate(flat)=%d"
          % (n_trials, checked, degenerate))
    print("  max |round(peak,3)-ref_peak| = %.4g   lag mismatches = %d"
          % (max_dpk, lag_bad))
    group("A. ac_limited == detector.ac_score on lag[%d,%d]"
          % (LAG_LO, LAG_HI), checked >= 500 and lag_bad == 0
          and max_dpk == 0.0)


def verify_clean_train():
    n = 210

    periodic = (np.arange(n) % 7 < 2).astype(np.uint8)
    peak, lag, curve = ac_curve(periodic, 5, 25)
    ok, lags = clean_train(curve, 5)
    print("  periodic p7 duty=%.2f peak=%.3f@%d -> %s lags=%s"
          % (periodic.mean(), peak, lag, ok, lags))
    c_periodic = ok is True and len(lags) >= 3

    block = np.zeros(n, np.uint8)
    block[60:90] = 1
    peak, lag, curve = ac_curve(block, 1, 40)
    ok, lags = clean_train(curve, 1)
    print("  single ON block duty=%.2f peak=%.3f@%d -> %s lags=%s"
          % (block.mean(), peak, lag, ok, lags))
    c_block = ok is False

    rng = np.random.default_rng(7)
    false_rate = 0.0
    trials = 200
    for _ in range(trials):
        nn = int(rng.integers(80, 400))
        p = float(rng.uniform(0.05, 0.7))
        on = (rng.random(nn) < p).astype(np.uint8)
        _, _, curve = ac_curve(on, 1, 40)
        ok, _ = clean_train(curve, 1)
        false_rate += (ok is False)
    false_rate /= trials
    print("  uniform random binary: False rate = %.3f over %d series"
          % (false_rate, trials))
    c_random = false_rate >= 0.9

    t = np.arange(n)
    envelope = np.linspace(0.0, 1.4, n) + 0.25 * np.sin(2 * np.pi * t / 50.0)
    high = (envelope > 0.15).astype(np.uint8)
    peak, lag, curve = ac_curve(high, 1, 40)
    ok, lags = clean_train(curve, 1)
    print("  high-duty slow ramp/plateau duty=%.2f peak=%.3f@%d -> %s lags=%s"
          % (high.mean(), peak, lag, ok, lags))
    c_high = ok is False

    group("B. clean_train synthetic checks",
          c_periodic and c_block and c_random and c_high)


def verify_onsets():
    cases = []

    a = np.zeros(10, np.uint8)
    cases.append(("all zeros", a, 0))

    a = np.ones(10, np.uint8)
    cases.append(("all ones", a, 1))

    a = (np.arange(10) % 2).astype(np.uint8)          # 0,1,0,1,...
    cases.append(("alternating 0101 (n=10)", a, 10 // 2))

    a = (1 - np.arange(10) % 2).astype(np.uint8)      # 1,0,1,0,...
    cases.append(("alternating 1010 (n=10)", a, 10 // 2))

    a = np.array([1, 0, 0, 0, 0], np.uint8)
    cases.append(("single pulse at start", a, 1))

    a = np.array([0, 0, 1, 0, 0], np.uint8)
    cases.append(("single pulse in middle", a, 1))

    ok_all = True
    for name, a, expected in cases:
        got = onsets_of(a)
        good = (got == expected)
        ok_all = ok_all and good
        print("  %-24s expected=%d got=%d %s"
              % (name, expected, got, "ok" if good else "MISMATCH"))
    group("C. onsets_of", ok_all)


def main():
    print("verify_periodicity: independent checks "
          "(sources under test are not modified)")
    verify_equivalence()
    verify_clean_train()
    verify_onsets()
    if FAILS:
        print("\nRESULT: FAIL (%d group(s)): %s" % (len(FAILS), ", ".join(FAILS)))
        sys.exit(1)
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
