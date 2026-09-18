# Overnight State Inventory + Budget

状态盘点与预算，先于 C3 window 选型与 `TemporalSeries` 改造。
所有数字来自代码与 `benchmark/out/` 实测，非估计处标注公式。

- 单位：MiB = 1024²。
- 代码基线：`deploy/app/night_lamp/*`、`deploy/app/server/*`（`deploy/` 为生产副本）。
- 实测来源：
  - `benchmark/out/soak_1749_20260918_161947.json`（4500 obs）
  - `benchmark/out/baseline_1749_pipeline_20260918_163508.json`（3747 obs）
  - `benchmark/out/periodicity_window_probe.json`（3000 samples）
  - `benchmark/out/ffmpeg_20260918_155020.json`
  - `benchmark/out/baseline_1749_micro_20260918_154533.json`

---

## 0. 固定 workload（预算前提）

| 项 | 值 | 来源 |
|---|---|---|
| 分辨率 | 1920×1080 | `tmp/*.mp4`，micro shape |
| observation | 6.25 Hz（interval_ms=160） | `deploy/config.yaml:66` |
| 夜长 | 7 h | 用户锁定 |
| `max_candidates` | 200 | `deploy/config.yaml:67` |
| `baseline.frames` | 60 | `deploy/config.yaml:68` |
| 算法语义 | 当前 `NightSession` 为准，不改 threshold/sampling/baseline/periodicity | — |
| 一晚 samples | `7*3600/0.160 = 157,500` | 推导 |

> 冻结：任何为换内存而改算法参数的做法，不在本预算内。

---

## 1. Overnight State Inventory

`Exact/Approx` 指该 state 对 morning report 是精确证据，还是可交换/可丢弃的近似。  
`Consumer` 决定“能不能丢”，而不是大小决定。

| State | Producer | Consumer | Lifetime | 当前大小 @1920×1080 | 随夜长增长 | Exact/Approx | Morning report 需要 |
|---|---|---|---|---|---|---|---|
| decoder frame slot `Reader._frame` (`server/source.py:29,144`) | RTSP/ffmpeg | `CameraWorker._loop` | frame | 1×BGR = 5.9 MiB（`read()` 每次再 copy 5.9 MiB） | 否 | exact | 否 |
| worker `_latest_frame` (`server/worker.py:252`) | CameraWorker | `finalize` base frame | whole night | 5.9 MiB | 否 | exact | 是（day 底图/对齐） |
| warmup `_warm_stack` (`session.py:393`) | observation | `_finish_warmup` | warmup only | 60×1080×1920 u8 = **118.7 MiB** | 否（结束后 `=None`） | exact | 间接（baseline） |
| warmup `_warm_meds` | observation | `_finish_warmup` | warmup | 60 floats | 否 | exact | 否 |
| `_base` (`session.py:351`) | warmup | `_process` / finalize | whole night | f32 = 7.9 MiB | 否 | exact | 是（阈值/dim_map） |
| `_bg` (`session.py:352`) | warmup | finalize render/align | whole night | f32 = 7.9 MiB | 否 | exact | 是（白天叠加） |
| `_on` (`session.py:353`) | observation | discovery + finalize | whole night | f32 = 7.9 MiB | 否 | exact | 是（duty map） |
| `_peak` (`session.py:354`) | observation | discovery + finalize | whole night | f32 = 7.9 MiB | 否 | exact | 是（reflection/host） |
| `_mask` OSD (`session.py:343`) | observation | `prep` | whole night | bool = 2.0 MiB | 否 | exact | 否 |
| `_meds` per-sample medians (`session.py:356,387`) | observation | `_night_qualification` | whole night | 157,500 py-float ≈ 4.8 MiB | **是** | exact | 是（night gate 标签） |
| `TemporalSeries._buf` (`session.py:90`) | observation | finalize periodicity | whole night | **400.0 MiB**（见 §2.1） | **是** | exact | 是（flash 判定核心） |
| `TemporalSeries` meta (`session.py:83-85`) | discovery | finalize | whole night | <50 KiB | 否 | exact | 是 |
| `CandidateDiscovery` cells/counters (`session.py:239-250`) | discovery | report metadata | whole night | O(candidates+dyn) ≈ 固定 | 否 | approx（dedup 半径） | 是（metadata） |
| discovery transient (split_comps / nonzero) | discovery | discovery pass | per-pass | 数十 MiB f32/int32 峰值 | 否 | exact | 否 |
| acq→obs queue | acquisition | observation | — | **无队列，depth=1 latest-wins**；无 drop 计数 | 否 | approx（静默丢帧） | 否 |
| `Reporter._q` (`reporting.py:70`) | finalize / day loop | callback thread | process | `maxsize=64`，每条含 JPEG base64（可达 ~64 MiB） | 否 | exact（满则丢 newest） | 是（投递） |
| `Reporter._last/_alarm_interval` | events | `submit_event` | process | O(cameras×algos) | 否 | exact | 否 |

### 1.1 语义消费者结论（决定能不能丢）

- `_base/_bg/_on/_peak/_mask`：fixed-size，report 必需，**不可丢**，但不是增长源。
- `TemporalSeries._buf`：report 的 periodicity 决策直接读取，**不可丢**；问题在“全历史 vs 有界窗口”，不在能否丢弃。
- `_meds`：仅供 `_night_qualification` 回放（`session.py:550`）。night gate 是**纯标量状态机**（state/count/transitions），无需全序列 → 可 online 化，属可消除的增长。
- `Reader._frame`：latest-wins，consumer 只看“当前”，可丢旧帧。
- discovery 状态：只决定“谁被跟踪”，不证明等价（`session.py:21-22`），可近似、可丢。

> 一句话：**能随夜长增长的只有两处 —— `TemporalSeries._buf`（主）与 `_meds`（次）。**

---

## 2. Overnight Budget

### 2.1 RSS（每相机）

```
samples(7h)          = 157,500
series.max_points    = 3*max_candidates + DYN_MAX_PEAK = 3*200 + 200 = 800   (session.py:334)
series.cols          = 2^ceil(log2(157500)) = 262,144
series._buf bytes    = 800 * 262,144 * 2 (int16) = 419,430,400 B
                     = 400.0 MiB   (steady)
grow transient peak  = new(400.0) + old(200.0) = 600.0 MiB  (131072 -> 262144)
```

实测验证：4500 obs 时 `series_end_mb = 12.5`。公式 `800*8192*2/1MiB = 12.5`。吻合，公式可外推。

| 项 | 当前 @7h | 性质 |
|---|---:|---|
| Python/native base（imports, cv2, numpy） | ≈ 46.8 | 实测 `soak rss_start_mb` |
| decoder / frame（含 read copy） | 5.9 + 5.9 | 稳定 + 瞬态 |
| warmup transient | 118.7 | 峰值，warmup 后释放 |
| spatial accumulator（base/bg/on/peak/mask） | 33.6 | 固定 |
| candidate / discovery state | <1 | 固定 |
| `_meds` | 4.8 | **单调增长** |
| `TemporalSeries` | 400.0 | **单调增长（blocker）** |
| series grow transient | 600.0 | 峰值 |
| Reporter queue（64×JPEG base64） | ≤ ~64 | 有界，极端 |
| **稳态合计（不含 transient）** | **≈ 491** | vs `memory_budget_mb_per_camera=500` |
| **瞬态峰值（warmup / series grow）** | **≈ 600+** | 远超 500 |

**结论：当前 `series_cap=None` 在 7h 时稳态已 ≈ 预算上限，增长瞬态 ~600 MiB 直接击穿 500 MiB/camera。这是硬 blocker。**

补充：`memory_budget_mb_per_camera` 目前**只被 config 校验解析，无运行时强制执行**（仅 `server/config_validate.py` 引用）。即超预算不会被发现。

### 2.2 CPU（每 160 ms slot）

| 阶段 | 实测 | 每 slot 折算 |
|---|---:|---:|
| observation accumulate | 15.04 ms/sample（process 7.41，series 0.11，其余 7.6） | ≈ 15.0 ms |
| discovery（每 50 samples 一次） | p50 35.96 / p95 42.32 / p99 51.58 / max 52.54 ms；pipeline 后段 65–80 ms | ≈ 1.3 ms（amortized） |
| periodicity | finalize 内 14.67 ms @ ~3.7k 长度 | 一次性 |
| report/finalize | 807 ms 一次 @ ~600 s 历史 | 一次性 |
| **slot 合计** | **≈ 16.5 ms / 160 ms** | ≈ 10%，机器相关 |

风险点：

1. 上表来自本机（较空闲）。discovery p99 52 ms + accumulate 落在同一 slot 时 ≈ 67 ms，仍 <160 ms，但余量取决于部署机。
2. `finalize` 的 periodicity 对每个 tracked row 调 `ac_limited(on, lag_lo, lag_hi)`，复杂度 **O(series_length × lag_range)**，随夜长线性。当前 600 s→14.67 ms；7 h 线性外推 ≈ 0.6 s。绝对不大，但**违反 bounded-runtime 约束**，且随 tracked rows 增长。→ 与 C3 的“有界 AC 窗口”是同一件事。
3. `_night_qualification` 在 finalize 用 Python 循环回放 157,500 个 median（`session.py:550`）。

### 2.3 Runtime stability

| 指标 | 现状 | 缺口 |
|---|---|---|
| queue max depth（acq→obs） | 无队列，latest-wins depth=1 | 无 drop 计数；ON/OFF 边沿可能被静默丢弃 |
| queue max depth（reporter） | `maxsize=64`，满则丢 **newest** | finalize report 被丢 = 无热力图，且只留限流 warning |
| allowed observation drop rate | 无显式指标；`NightAdapter` 按 wall clock 门控有隐式丢帧 | 未量化、未上报 |
| effective sampling rate | 实测 6.2436 Hz（target 6.25） | 已上报 `sampling.actual_rate_hz` |
| discovery latency p50/p95/p99 | 35.96 / 42.32 / 51.58 ms | 未纳入 report metadata 的稳定性断言 |
| reconnect / stall | `RETRY_WAIT_S=3.0`，`TIMEOUT_US=10s` 重连 | 重连期间 observation 空窗未计数 |
| candidate capacity / overflow | `max_candidates=200`；**4500 obs 时已 `candidate_overflow=true`**（`n_cand_total=204`，online dropped 3730） | 12 分钟片段就溢出 → 容量/去重策略需重估 |

> candidate overflow 在 12 分钟素材就已发生，说明 200 这个上限对当前 discovery 过于激进；这是**语义**问题（丢的是候选），不是内存问题。

### 2.4 硬约束（本轮判定规则）

> **任何随 overnight sample count 单调增长的 state，一律视为未满足 bounded-runtime。**

据此：

- `TemporalSeries(cap=None)` → **主 architectural blocker**（§2.1，稳态 400 MiB / 瞬态 600 MiB）。
- `_meds` → 次 blocker（可 online 标量化，成本低）。
- spatial accumulators（fixed-size）→ 不误伤，保持 exact。
- queue → 已是 depth=1，但缺显式 drop 策略与计数。

---

## 3. C3 验证：与 Budget 解耦

**不在此选 128/256/512。** 先建立保真度契约，再反推最小 temporal state。

```
golden set ──► reference(full history) ──► bounded candidate ──► fidelity metrics
```

- reference = `cap=None` 的全历史决策，作为**语义基线**，不是 ground truth。
- bounded candidate = `(W, onset_policy)`，W 为尾部窗口，`onset_policy ∈ {naive truncate, online full count}`。
- 输出：满足契约的**最小** `(W, onset_policy)`。

### 3.1 覆盖矩阵（每格都要有 golden）

| 维度 | 取值 |
|---|---|
| period | 0.56 s / 1.0 s / 2.2 s |
| phase | 至少 4 相位（相对采样格点） |
| duty cycle | 低 / 中 / 高 |
| flash 强度 | weak / strong |
| candidate emergence | 早期 / 中段 / 临近 07:00 |
| observation drop | 0% / 5% / 20% |
| jitter | 0 / ±20% interval |
| missing observations | 连续缺 k 个 |
| candidate churn | 稳定 / 频繁出现-消失 |
| sampling rate 边界 | 5.5 / 6.25 / 7.0 Hz |

### 3.2 指标（对每格报告，不取平均掩盖）

- per-row `flashing` 混淆：agreement / false-neg / false-pos（对 reference）。
- report 级：`n_flash`、rings、`n_cand` 一致性。
- emergence time 误差（秒）。
- 契约形式：**所有格 ≥ 阈值，且 FN/FP 有上界**；不是“整体 recall 99%”。

### 3.3 已有先验（不得作为验收）

`periodicity_window_probe.json`（3000 samples，单素材，len≤2750）：

| W | A(naive) agree | B(online onset + windowed AC) agree |
|---|---|---|
| 128 | 0.730 | 0.9862 |
| 256 | 0.9613 | **0.9900** |
| 384 | 0.9838 | 0.9875 |
| 2048 | 0.9938 | 0.9938 |

`onset_gate_flips` 从 W=32 的 234 降到 W=2048 的 6 → 印证：**onset 计数应 online 保留，只窗口 AC**。  
但该结果是单一条件、短历史，**不能作为 production 验收**。

---

## 4. 目标 Runtime 架构

```
                 ┌───────────────┐
RTSP ───────────►│ Acquisition   │  server/source.py Reader（depth=1 latest-wins）
                 └───────┬───────┘
                         │ bounded queue (depth 1, 显式 drop 计数)
                         ▼
                 ┌───────────────┐
                 │ Observation   │  NightAdapter（160ms gate）+ NightSession.accumulate
                 └───┬───────┬───┘
                     │       │
              spatial│       │temporal
                     ▼       ▼
              fixed-size   bounded
              accumulator  temporal state   ◄── 当前缺口：cap=None 无界
                     │       │
                     └───┬───┘
                         ▼
                    Morning report
                         │
                    discard/reset（release，按夜隔离）
```

**queue 必须 bounded**：当前 acq→obs 不是队列而是 depth=1 最新帧，已天然有界；但缺显式 drop 策略。否则一旦引入缓存，acquisition 快于 processing 就会变成另一种“存 overnight 数据”。

Drop policy（需明确，不能只有全局 latest-wins）：

- spatial evidence：可较 aggressive 降采样（统计量，容忍近似）。
- temporal evidence：必须保留短 ON/OFF transition（周期判定依赖边沿）。
- queue overflow：明确“哪些 observation 可牺牲、哪些 temporal evidence 优先”。

---

## 5. 工程顺序（固定）

**A State Inventory → B Budget → C3 fidelity/最小 window → D bounded runtime architecture → E real-RTSP overnight soak**

不是：再优化几毫秒 → FakeClock 7h → 宣称 overnight 完成。

### 立即可做（按性价比）

1. 本文件：Inventory + Budget 基线（完成）。
2. `_meds` online 化：night gate 只需 (state, count, transitions) 三个标量，删掉 4.8 MiB 单调增长 + finalize Python 循环。
3. C3 golden set + fidelity harness（§3），用于选最小 `(W, onset_policy)`。
4. 用 C3 结果替换 `series_cap`，并加 `max_points/cols` 预算断言。
5. D：显式 drop 计数与 transition-preserving 策略；queue 保持 bounded。
6. E：real-RTSP 7h soak，校验 §2.3 全部指标。

---

## 6. Overnight Sampling Study（实测，mini-night）

harness：`benchmark/sampling_study.py`（benchmark-only，复用生产 `NightSession` + `nightly_map` 纯函数）。
素材：`tmp/1750_20260913_220000.mp4`（35 min，1920×1080，6.25 Hz）。
reference：全速率 13,112 obs，124 candidates，37 flashers。
结果 JSON：`benchmark/out/sampling_study_1750_20260918_171923.json`。
口径：spatial 与 schedule 有关；temporal 分 persistent（单 session）与 reset（观测间隔 >30 s 重置、取 burst 并集）。

| policy | obs% | duty_corr | hotspot IoU | P.recall | P.prec | R.recall | R.prec |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 99.7% | 1.000 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| burst 30s / 5min | 10.0% | 0.974 | 0.674 | 0.86 | 0.89 | **0.00** | – |
| burst 60s / 5min | 20.0% | 0.984 | 0.742 | 0.92 | 0.83 | **0.00** | – |
| burst 120s / 5min | 40.0% | 0.993 | 0.827 | 0.92 | 0.81 | 0.84 | 0.91 |
| burst 60s / 10min | 11.4% | 0.973 | 0.680 | 0.78 | 0.73 | **0.00** | – |
| burst 120s / 10min | 22.9% | 0.975 | 0.745 | 0.89 | 0.85 | 0.84 | 0.91 |
| random 10% | 10.3% | **0.999** | **0.921** | **0.05** | 0.33 | **0.05** | 0.33 |
| sparse 4×120s | 22.9% | 0.979 | 0.754 | 0.89 | 0.79 | 0.84 | 0.89 |

### 结论（cold）

1. **Spatial 不是问题。** 连 random 10% 的 duty_corr=0.999、IoU=0.92。heatmap 对 observation 数极不敏感。
2. **Temporal 才是约束，且必须 burst。** random 10% 空间最好，但 flasher recall **0.05**；burst 20% 达 0.89–0.92。→ burst sampling 假设成立；sparse individual 采样对 periodicity 无效。
3. **reset 需要 burst ≥ ~120 s。** 30/60 s burst 的 reset recall = 0，被 `_MIN_DISCOVERY_SAMPLES=250`（~40 s）+ `onset_min=50` 卡死：burst 内可跟踪序列太短，onset 数不够。
4. **persistent 可救 60 s burst**（recall 0.78–0.92），且跨 burst 的相位不连续不是主要误差（peak lag dev = 0）。主要误差来自 discovery warm-in 与 onset_min，而非相位。
5. **代价曲线**：35 min 上 10%/20%/40% → 22/44/88 s CPU（proj），continuous 218 s。线性外推 7 h：continuous ~44 min，20% burst ~9 min，10% ~4.5 min（**仅算法；decoder 若常开不省**）。
6. **精确性上限 ~0.92**，非 1.0：重采样引入 FP/FN。验收阈值必须显式，不能默认“接近 1”。

> harness 自身同时持有 reference + 8 个 policy session，RSS 峰值 2.2 GB，**是 benchmark artefact，不是生产占用**。生产一次只跑一个 schedule。

### 对 C3 的影响

C3 从“7 h 全局历史怎么压”缩小为“**单 burst 内 tracking 需要多少 temporal state**”，上界 = burst 长度（120 s ≈ 750 samples），比 157,500 小两个数量级。但 `onset_min=50` 与 burst 长度的耦合必须进 fidelity 契约：短 burst 要么保 online onset 计数，要么重设 `onset_min`。

---

## 附：口径

- `series_cap` 语义（`server/contracts.py:93`）：`None`=全序列；非 None=尾部窗口，实验性。
- `max_candidates` 溢出（`session.py:296,617`）：置 `candidate_overflow`，`Report.status=degraded`，绝不静默取前 N。
- 本表 RSS 为 per-camera；多相机需线性叠加（另加进程共享的 cv2/numpy base 一次性开销）。
