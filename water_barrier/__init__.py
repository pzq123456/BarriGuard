"""Water-barrier gap alarm algorithm (core detection library).

The server/ package is the realtime inference framework (capture, temporal
confirmation, HTTP). This package is the alarm algorithm it calls per frame:

    from water_barrier import BarrierAlarm
    alarm = BarrierAlarm()          # construct once per stream
    gaps = alarm.step(frame_bgr)    # per frame -> list[Gap] with pixel boxes

Public surface: alarm.BarrierAlarm, alarm.Gap, alarm.detect_frame.
Offline validation entry: python -m water_barrier.main
"""

from .alarm import BarrierAlarm, Gap, detect_frame

__all__ = ["BarrierAlarm", "Gap", "detect_frame"]
