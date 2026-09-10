# 实验报告 R4：Soak 烧机验证 LED 烧屏思路 + 上线配置（2026-09-10）

## 烧屏思路是什么（用户原话：类似于 LED 烧屏）

LED 烧屏/老化筛选：让被测对象持续通电跑一段时间，偶发毛刺自己消失，
真缺陷一直都在——**用时间换可靠性**。对应到本系统：transient（人/车/眩光）
靠 `intact_reset/track_stale` 时间衰减自消，永久缺口熬过 `alarm_hold_s`
才报警。R4 用 1749 真实时序（`tmp/cap_day`，465 帧/87 分钟，~11s 间隔）
验证这条机制是否成立。

## Soak 原始结果（`python -m water_barrier.soak`，三档 hold）

```
hold=10 : alarm_frames=464 alarm_episodes=1 first_alarm_t=10  evidence=17 status={SUNGLARE:437, OK:28}
hold=60 : alarm_frames=459 alarm_episodes=1 first_alarm_t=60  evidence=16 status={SUNGLARE:437, OK:28}
hold=300: alarm_frames=435 alarm_episodes=1 first_alarm_t=307 evidence=16 status={SUNGLARE:437, OK:28}
```

框抽查（6 个时刻）：同一个门洞框 `(~815,449,904,651)` 全程稳定。
单帧耗时：mean 434ms / p95 605ms（约 2.3fps，Reader 只留最新帧故不积压，
静态场景够用；要更高帧率需另做性能项）。

## 结论（直接回答上线三问）

1. **机制成立**：87 分钟真实流只有 1 个 episode——固定门洞。hold 从 10 拉到 300，
episode 数不变（永久缺口熬得住），首报延迟 = hold（10/60/307s），transient
（87 分钟内的行人车辆波动）一个都没熬成 alarm。烧屏筛选有效。
2. **报警频率可限可调**，两处旋钮（都已配进 yaml，不用改代码）：
   - 检出延迟：`track.alarm_hold_s`（静态场景建议 60；等得起就 300，误报→0 的方向）
   - 落盘频率：`server.evidence_resave_s`（新配进 `server/config.yaml`，默认 300；
   本次 87 分钟持续报警只存 16~17 张：上升沿 1 张 + 每 5 分钟 1 张）
3. **静态场景推荐配置**：`alarm_hold_s: 60`（1 分钟延迟换 transient 全灭）、
`intact_reset_s: 5`（遮挡离开 5 秒即脱锁，不拖尾）、`evidence_resave_s: 300~600`。
`intact_reset/track_stale` 保持默认即可，不要动（动了会 laten 真缺口恢复）。

## 能否上线（诚实版）

能，但三条边界写进运维手册：①只验了日间（夜间/阴雨未验，floor 0.24 与影白
分档同理）；②1750 日光时段真缺口灵敏度≈0（R3 问题 3，信号物理决定），该路
日间只能保“零误报”不能保“必检出”，靠 SUNGLARE 状态显式标记；
③单流约 2.3fps，两路并发约翻倍 CPU，机器扛不住再谈优化/降帧。

## 核心代码（已进仓库，不在 /tmp）

烧机核心（`water_barrier/soak.py:run_soak`，与框架无关，只认 Algorithm 协议）：

```python
res = algo.step(frame, float(t))
alarming = any(e.kind == "alarm" for e in res.events)
...
if t - last_save >= resave_s:   # 上升沿 + 节流，和线上 EvidenceWriter 同规则
    saves += 1
```

落盘节流接线（`server/app.py` + `server/config.yaml`）：

```python
_evidence = EvidenceWriter(resave_s=_p.get("evidence_resave_s", 300.0))
# evidence_resave_s: 300.0  # 同一次持续报警的补存间隔（静态场景可加大）
```

粗糙度实验同步搬入 `water_barrier/roughness.py`
（`python -m water_barrier.roughness --img ... --row 0`，含 R2 踩坑注释：
未平滑像素+口径对齐）。
