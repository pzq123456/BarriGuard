"""BarriGuard 服务包：RTSP 拉流 -> 缺口报警 -> 时序确认 -> 叠加展示。

与 tmp/server 的区别：去掉固定 slot（GAP/NEAR/MID/FAR 预标记细框），缺口由
water_barrier.BarrierAlarm 自主检出，再经 server/track.py 时序确认。
"""
