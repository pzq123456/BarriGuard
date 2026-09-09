# 夜间警示灯抽查频率（实验结论）
依据：`burst_test.py` 全帧率实测（1749 机位，12.49fps，6盏代表灯）。

## 实测界（本站）

* 闪光周期 `T ≤ 2.2s`，占空比 `d ≥ 13%`，最大 OFF 间隙约1.9秒。
* ON 绝对规则：`7x7 max ≥ 中值+40 或 mean ≥ 中值+30`（自适应分位数会对静态罩编造 duty，禁用）。

## 巡检频率：每小时 1 次，每次连续 3.2 秒（40帧）

| 突发长 | 强灯检出 | 弱灯检出 | 死灯误报 |
|---|---|---|---|
| 0.32s | 0.72 | 0.46 | 0 |
| 0.8s | 1.0 | 0.92 | 0 |
| 2.4s+ | 1.0 | 1.0 | 0 |

30轮模拟：闪光灯 30/30，死灯/反光片 0/30。理论式 `P=min(1,(S+d·T)/T)`。

## 三条铁律

1. **必须连续突发，禁止固定周期散抽**：步长撞上周期时检出率掉到0.57（1263灯，步长7帧）。
2. **判闪≥1个ON，加固版≥2个ON**（死灯全片仅3噪点帧）；SUSPECT连中两次才告警。
3. **换机位重估 `T/d` 再定突发长**：突发 ≥ 最大OFF间隙+1秒余量；遮挡帧报 UNKNOWN 不判死。

## 生产运行（Phase 1，已抽取未优化）

```bash
python main.py --config configs/config_1749.yaml --mode dry    # 1 burst 验证
python main.py --config configs/config_1749.yaml --mode full   # N bursts，见 burst.count
python tools/inspect.py --out output/production_1749  # 查结果
```

冷启动标注在 `configs/registry_1749.csv`（Excel 可编辑，kind=lamp/watchlist）；
冻结背书（version/frozen/count）在 `configs/config_1749.yaml` 的 registry 块，loader 强制校验。

detector = overnight 冻结版（`detector.py`，行为与 `overnight_run.py` 一致，已回归）。
night_state = P1 provisional（`enter_threshold=100/exit_threshold=120/p=2`，配置化；
单夜推导的生产起点，非通用验证阈值）。
`require_night_for_detection=true`：非 NIGHT burst 跳过 detector（gated，可记录、可回放验证）。

## 工程化原则（KISS）

1. 一个摄像头 = 一个 YAML config。
2. Camera-specific 信息放 config/registry，不复制代码。
3. 只有存在真实跨调用状态时才引入 state（目前仅 night qualification）。
4. 只有存在真实复用需求时才抽象函数/类。
5. 实验代码和生产代码物理分开，不把实验框架带进 runtime。
6. Don't abstract for hypothetical camera #2：第二个摄像头首先只是 `config_1750.yaml` + `registry_1750.yaml`。
