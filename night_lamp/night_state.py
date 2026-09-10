"""Night qualification state -- EXTRACTED from analyze_night_qualification.py.

P1 provisional (single night, 9 bursts): T_enter=100, T_exit=120, p=2.
Hysteresis required: T_enter < T_exit. Recalibrate with production data.

update() is the single cross-burst step, persisted by main.py in
night_state.json -- the only cross-burst state.
"""
NIGHT_VERSION = "p1-provisional-20260908"

TWILIGHT = "TWILIGHT"
NIGHT = "NIGHT"


def update(state, count, g, t_enter, t_exit, persistence):
    """One step. Returns (new_state, new_count, entered_bool)."""
    entered = False
    if state == TWILIGHT:
        count = count + 1 if g < t_enter else 0
        if count >= persistence:
            state, entered = NIGHT, True
    else:
        if g > t_exit:
            state, count = TWILIGHT, 0
    return state, count, entered


def load(path):
    import json
    try:
        d = json.load(open(path, encoding="utf-8"))
        return d.get("state", TWILIGHT), int(d.get("count", 0))
    except (FileNotFoundError, ValueError, KeyError):
        return TWILIGHT, 0


def save(path, state, count):
    import json
    json.dump({"state": state, "count": count, "version": NIGHT_VERSION},
              open(path, "w", encoding="utf-8"))
