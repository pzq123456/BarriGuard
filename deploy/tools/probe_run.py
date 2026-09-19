"""Phase-1 probe driver: run real config paths + one full acceptance cycle.

Usage (from repo root, any cwd):
    <venv>/python deploy/tools/probe_run.py [--video <mp4>]

It installs the read-only probe, then:
  A. exercises config discovery + load_runtime (production and mini) and the
     legacy load() path, recording every file it touches;
  B. constructs the real night/water consumers from the loaded config;
  C. runs the deterministic acceptance suite (full night lifecycle + burst).
The JSON report is written by probe_config.dump().
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
DEPLOY = TOOLS.parent
REPO = DEPLOY.parent
APP = DEPLOY / "app"
TESTS = DEPLOY / "tests"

for _p in (str(APP), str(TESTS), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_config as probe  # noqa: E402


def phase_a_config(prod_cfg_path: Path):
    print("\n=== Phase A: config discovery / load ===")
    from server import config

    resolved = config._config_path()
    print(f"[A] _config_path() -> {resolved}")

    cfg = config.load_runtime(prod_cfg_path)
    probe.wrap_config_dicts(cfg)
    print(f"[A] load_runtime({prod_cfg_path.name}) -> {len(cfg.cameras)} cameras")

    removed = not hasattr(config, "load")
    for label in ("legacy load(None)", "legacy load(prod)"):
        outcome = "removed (config.load deleted)" if removed else "present"
        probe.record_path("server.config.load", label, None, [], outcome)
        print(f"[A] {label} -> {outcome}")
    return cfg


def phase_b_consumers(cfg, video: Path):
    print("\n=== Phase B: real consumers from loaded config ===")
    import cv2 as cv
    from night_lamp.adapter import NightAdapter
    from night_lamp.session import NightSession
    from server import registry, worker
    from server.schedule import RealClock

    cam = cfg.cameras[0]
    night_spec = cam.algorithms["night_lamp"].spec
    water_spec = cam.algorithms["water_gap"].spec

    # night: constructing the session reads night_gate; adapter reads sampling.
    clock = RealClock()
    session = NightSession(cam.id, night_spec)
    NightAdapter(session, night_spec, clock)
    print(f"[B] NightSession+NightAdapter built for camera {cam.id}")

    # night: go through worker's factory (builds NightAdapter from spec).
    try:
        worker._default_night_adapter(cam.id, night_spec, clock)
        print("[B] worker._default_night_adapter OK (calibration consumed)")
    except Exception as exc:  # noqa: BLE001
        print(f"[B] worker._default_night_adapter failed: {type(exc).__name__}: {exc}")

    # water: policy merge + real algorithm on one frame.
    calib = worker._load_calibration("water_gap", water_spec)
    calib = probe.wrap_dict(calib, "calib_merged[water_gap]")
    cap = cv.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read a frame from {video}")
    algo = registry.create("water_gap", frame.shape, calib, cam.id)
    res = algo.step(frame, 0.0)
    print(f"[B] WaterGapAlgorithm.step -> {len(res.events)} events, "
          f"{len(res.annots)} annots")
    session.release()


def phase_c_acceptance(video: Path):
    print("\n=== Phase C: acceptance suite (deterministic business cycle) ===")
    import replay_harness as H
    import acceptance

    results = acceptance.run_all(H.Context(video=str(video)))
    for r in results:
        print(f"[C] [{r.status:10}] {r.name}  {r.detail}")
    print(repr(H.format_results(results)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO / "tmp" / "1002490_night_10min.mp4"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    prod_cfg = DEPLOY / "config.yaml"

    probe.install()
    cfg = phase_a_config(prod_cfg)

    phase_b_consumers(cfg, video)
    phase_c_acceptance(video)

    out = Path(args.out) if args.out else probe.default_out()
    written = probe.dump(out)
    snap = probe.snapshot()
    print(f"\n=== REPORT: {written} ===")
    print(f"fields_used={len(snap['fields_used'])} "
          f"dict_tags={len(snap['dict_keys_used'])} paths={len(snap['paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
