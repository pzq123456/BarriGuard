"""Night heatmap Report assembly (Wave 1, Agent D).

Single place for the Report shape and the metadata vocabulary used by
``NightSession.finalize``.  Keeping the strings here means ``session.py`` and
``adapter.py`` never hardcode report types or statuses, and the Reporter
(Agent E) has a frozen contract to consume.

Only ``server.algo`` (Report) and ``server.contracts`` (AlignmentStatus) are
imported; ``nightly_map`` / ``periodicity`` are untouched.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.algo import Report  # noqa: E402
from server.contracts import AlignmentStatus  # noqa: E402

ALGO_NAME = "night_lamp"
REPORT_TYPE = "night_heatmap"          # 早晨最终热力图
REPORT_TYPE_BURST = "night_heatmap_burst"  # 每小时 burst 累计热力图
JPEG_QUALITY = 80

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"

M_ALIGNMENT = "alignment_status"
M_CANDIDATE_OVERFLOW = "candidate_overflow"
M_SAMPLING = "sampling"
M_N_CAND = "n_cand"
M_N_CAND_TOTAL = "n_cand_total"
M_N_FLASH = "n_flash"
M_N_REFLECT = "n_reflect"
M_ONLINE = "online_candidate_discovery"
M_DAY_ALIGN = "day_alignment"
M_IMAGE_MODE = "image_mode"
M_NIGHT_MED = "night_med"
M_DELTA = "delta"
M_REASON = "reason"
M_WARMUP = "warmup"
M_MEMORY_MB = "memory_estimate_mb"
M_LAG_FRAMES = "lag_frames"
M_PERIOD_STEP = "period_step"
M_BURST = "burst"
M_ADAPTER = "adapter"
M_RESTORED = "restored"
M_FLASH_SEMANTICS = "flash_semantics"
M_FLASH_AREA = "flash_area"
M_FLASH_UNION_AREA = "flash_union_area"
M_FLASH_BUCKET_COUNT = "flash_bucket_count"
M_CALIBRATION_STATUS = "calibration_status"
M_LAMP_ROI = "lamp_roi"

# Frozen semantics: the morning map is the OR of per-hour flash presence,
# never a sustained-duration map (see the Phase-0 decision).
FLASH_SEMANTICS_PRESENCE = "presence_or"   # morning: OR of hourly presence
FLASH_SEMANTICS_BURST = "burst_periodic"   # hourly: that burst's periodic verdict

IMAGE_MODE_NIGHT = "night"
IMAGE_MODE_DAY = "day_overlay"


def build_report(camera_id, created_at, image_jpeg, metadata, status=STATUS_OK,
                 report_type=REPORT_TYPE):
    """Assemble the frozen Report dataclass for a night heatmap."""
    return Report(camera=camera_id, algorithm=ALGO_NAME, report_type=report_type,
                  created_at=created_at, image_jpeg=image_jpeg,
                  metadata=metadata, status=status)


def alignment_status_for(cc, alignment_spec):
    """Map the ECC correlation to AlignmentStatus using the config policy.

    ``align_day`` returns ``cc = -1.0`` when registration fails outright; that
    is REJECTED, never a silent no-op.  Below ``min_cc`` the configured
    ``on_low_cc`` applies (degraded by default).
    """
    if cc is None or cc < 0:
        return AlignmentStatus.REJECTED
    if cc < float(alignment_spec.min_cc):
        return alignment_spec.on_low_cc
    return AlignmentStatus.ALIGNED


def status_for_alignment(status, alignment_status):
    """REJECTED alignment must not publish a normal-looking day overlay."""
    if alignment_status == AlignmentStatus.REJECTED:
        return STATUS_REJECTED
    if alignment_status == AlignmentStatus.DEGRADED and status == STATUS_OK:
        return STATUS_DEGRADED
    return status
