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

## 结论（直接回答上线三问，R5 修订版）

1. **机制成立，但 R4 的证据链有缺口（见 R5）**：87 分钟 soak 全程 suspected=0——
窗口里根本没出现过 transient，episode=1 只证明了“真缺口扛得住 hold”，
没证明“hold 拦得住 transient”。真正的拦截证据见 R5 货车序列对照。
2. **报警频率可限可调**，两处旋钮（都已配进 yaml，不用改代码）：
   - 检出延迟：`track.alarm_hold_s`（当前默认 10；60/300 为可选项，见下）
   - 落盘频率：`server.evidence_resave_s`（新配进 `server/config.yaml`，默认 300；
   本次 87 分钟持续报警只存 16~17 张：上升沿 1 张 + 每 5 分钟 1 张）
3. **静态场景配置（修订）**：`alarm_hold_s` 保持默认 **10**；60 为可选项，
**需安防运营方确认 60 秒延迟可接受后才可采用**（非技术单方决策）。
`intact_reset_s: 5`、`evidence_resave_s: 300~600` 可直接用。
`intact_reset/track_stale` 保持默认，不要动。

## 能否上线（诚实版）

能，但四条边界写进运维手册：①只验了日间（夜间/阴雨未验，floor 0.24 与影白
分档同理）；②1750 日光时段真缺口灵敏度≈0（R3 问题 3，信号物理决定），该路
日间只能保“零误报”不能保“必检出”，靠 SUNGLARE 状态显式标记；
③单流约 2.3fps，两路并发约翻倍 CPU，机器扛不住再谈优化/降帧；
④**停驶车辆是 hold 机制的天然盲区**：静止遮挡物在静态帧上与真缺口不可分
（R2 已证 rer 重叠），停留超 hold 必报，拉长 hold 无用，只能靠人工复核框图
（框内是车还是路一眼可辨）或后续加车型识别，不在本次解决范围。

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
