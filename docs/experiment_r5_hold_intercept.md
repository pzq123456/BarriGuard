# 实验报告 R5：hold 真的拦下过 transient 吗（2026-09-10）

回应专家对 R4 的统计批评：87 分钟 soak 的 suspected 全程为 0，
窗口里根本没有 transient，episode=1 只证明“真缺口扛得住 hold”，
没证明“hold 拦得住 transient”。本轮用 R2 已抓到的货车遮挡
（rer 0.577，静态与真缺口不可分）构造真实 transient 序列补证。

## 序列（dt=10s，1750 标定）

clean,clean → dahua_01×3（货车停留 30s）→ dahua_02 → dahua_03 → clean×3（车走）

## 原始输出：同一辆车，hold=10 漏报 40 秒，hold=60 全程无声

```
===== hold=10 =====
t= 20 suspected×6                              # 车进
t= 30 suspected×6 + (alarm, 0.564)             # 漏报
t= 40..70 (alarm, 0.564) 持续                  # 车已走（t≥50），拖尾~40s
t= 80 全 suspected，alarm 消失                 # intact_reset 脱锁
===== hold=60 =====
t= 20..90 suspected 若干，alarm 0 帧           # 30s 停留熬不过 60s 门，全程无声
```

拦截机制代码（`track.py`，累积不到门限就一直在 SUSPECTED 打转）：

```python
if self._accum < self._hold:     # 30 < 60，车走了都没攒够
    self.state = GapState.SUSPECTED
    return self.state
```

## 结论与诚实保留

1. hold 拦截 transient 是真的（同车对照，10 漏报 / 60 拦住），R4 缺的证据链补上。
2. 拖尾存在：hold=10 下车走后 alarm 又拖 ~40s（衰减 0.5/s + RER 迟滞），
线上看到“车走框还在”属正常收敛过程，不是 bug。
3. 天花板：**停驶超 hold 的车照样报**（静态不可分，R2 已证），hold 只筛“会走的”，
不筛“停着的”。此条已写入 R4 运维边界④。
4. 外部效度：仍是一天单天气，跨天/跨天气复测待补；60s 延迟待运营方确认（R4 已改）。
