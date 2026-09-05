"""BarriGuard 服务包：RTSP 拉流 -> 自主分割 -> 自主缺口检测 -> 时序确认 -> 叠加展示。

与 tmp/server 的区别：去掉固定 slot（GAP/NEAR/MID/FAR 预标记细框），缺口由
water_barrier/research/segment.py + detect.py 自主检出，再经 server/track.py 时序确认。
"""
