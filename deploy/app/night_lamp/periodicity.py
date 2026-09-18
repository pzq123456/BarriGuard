"""Periodicity features for long per-pixel series.

duty answers "how much" a pixel lit; it does not answer "how it lit". For
the nightly map the missing question is periodic flash vs one-off transit,
and that lives in the autocorrelation at non-zero lags.

`ac_limited` is the lag-limited, O(N*L) form of `detector.ac_score`. On a
whole-night series (tens of thousands of samples) the full `np.correlate`
in `ac_score` is O(N^2); here only the lags we care about (a few seconds)
are computed. The estimator and normalization are identical to
`detector.ac_score`, so the two agree on any shared lag window; if one
changes the other must follow.
"""
import numpy as np


def ac_limited(on, lag_lo, lag_hi):
    """Autocorrelation of `on` over lags [lag_lo, lag_hi].

    Returns (peak, lag, curve): `curve[i]` is the zero-mean, sum-of-squares
    normalized AC at lag `lag_lo + i`, matching `detector.ac_score`'s
    acf[L] = sum(x[:N-L]*x[L:]) / sum(x*x). Returns (0.0, -1, []) when the
    series is flat or the window is empty. O(N * (lag_hi - lag_lo + 1)).
    """
    x = np.asarray(on, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    e = float(x @ x)
    if e <= 0 or lag_lo < 0 or lag_hi < lag_lo or lag_lo >= n:
        return 0.0, -1, np.zeros(0, np.float64)
    lag_hi = min(lag_hi, n - 1)
    curve = np.empty(lag_hi - lag_lo + 1, np.float64)
    for i, lag in enumerate(range(lag_lo, lag_hi + 1)):
        curve[i] = float(x[:n - lag] @ x[lag:]) / e
    j = int(np.argmax(curve))
    return float(curve[j]), lag_lo + j, curve


def local_max_lags(curve, lag_lo, floor=0.2):
    """Lags of interior local maxima in `curve` that clear `floor`.

    Adjacent flat-top maxima are merged so a plateau counts once.
    """
    out = []
    for i in range(1, len(curve) - 1):
        if curve[i] >= curve[i - 1] and curve[i] >= curve[i + 1] \
                and curve[i] >= floor:
            if not out or (lag_lo + i) - out[-1] > 1:
                out.append(lag_lo + i)
    return out


def clean_train(curve, lag_lo, floor=0.2, min_peaks=3, gap_tol=1):
    """True when some run of `min_peaks` AC maxima is near-equal-spaced.

    The separating feature is peak *spacing consistency*, not peak height:
    a one-off transit decays monotonically (no interior maxima), a near
    steady high-duty point gives irregular long-lag maxima, a real flasher
    gives a train of equal gaps. Any consecutive run counts, so a fast
    flasher with an occasional missed peak is not thrown away. Returns
    (bool, lags).
    """
    lags = local_max_lags(curve, lag_lo, floor)
    if len(lags) < min_peaks:
        return False, lags
    for s in range(len(lags) - min_peaks + 1):
        gaps = np.diff(lags[s:s + min_peaks])
        if gaps.max() - gaps.min() <= gap_tol:
            return True, lags
    return False, lags


def onsets_of(on):
    """Number of 0->1 transitions (plus a leading 1). Cheap, no AC."""
    on = np.asarray(on, bool)
    if on.size == 0:
        return 0
    return int(np.count_nonzero(on[1:] & ~on[:-1])) + int(on[0])
