# experiments/ — finished research, evidence only

Scripts here ended with Phase 1. Moved verbatim, never edited afterward.
Do not import them from production (`main.py` depends on none of them).
Old outputs stay under `../output/` untouched.

## File map

* `overnight_run.py` + `stream_source.py` — overnight stability run (2026-09-08,
  9 bursts, RTSP). Frozen detector source: ON=`7x7max>=med+40|mean>=med+30`,
  A/B/C profiles. Production `detector.py` is its math, extracted unchanged.
* `analyze_night_qualification.py` — P1 analysis: T_enter=100 / T_exit=120 /
  persistence=2 from one night. Production `night_state.py` is its rule.
* `night_lamp_pipeline.py` — v3 full-video scan research pipeline (not burst
  based). Did NOT enter production.
* `blind_test.py` — zero-prior blind localization, front/back-half validation.
  Registry baseline evidence. Did NOT enter production.
* `localize.py`, `result_figs.py` — one-off auxiliaries of the v3 pipeline.

## P1 / P2 status

* P1 (night qualification): done, provisional thresholds in production config.
* P2-A/B/C0/D outputs: under `../output/p2_feasibility/` (figures + npy + json).
  Their driver scripts were never in this directory; only outputs are kept.
* P2-E (aggregation): verdict NOT PROMISING for single-step frame-local
  aggregation (see `../output/p2e_aggregation/report.md` §14–15). Stopped,
  no P2-F. This stop is why production keeps the overnight frozen detector
  instead of a small-window aggregation detector.

## Production boundaries that came from these experiments

* 40-frame burst / ~hourly cadence (`README.md` iron laws).
* Frozen ON rule + A/B/C (no DEAD/SUSPECT machine, no far-line redesign).
* Night gate provisional note in `configs/config_1749.yaml`.
