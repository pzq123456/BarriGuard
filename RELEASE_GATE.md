# BarriGuard Release Gate

上线风险 → 修复依赖 → 验收标准 的收敛清单。
范围：**不含算法本身逻辑**；鉴权/TLS 为**新需求，明确不做**。

计数口径：`P-c`、`P-d` 为两条独立 issue。

| 分类 | 数量 |
| --- | --- |
| P0 / P1 / P2 待修 | 20（其中 #3 已降级为忽略，故实际待修 19） |
| 业务决策项 | 1（#6） |
| 明确不做 | 1（#7） |
| 撤销项 | 3（#5、#11b、#4 数字） |

---

## 一、P0 — Release Blocker

P0 本质两类：**数据可靠性**（#1、#2）、**运行态可观测性/通知可靠性**（#3、#8）。

| # | 问题 | 影响 | 建议验收 | 状态 |
| --- | --- | --- | --- | --- |
| **1** | 夜间累积只存内存，20:00–07:00 重启后数据清零 | 直接导致整夜热力图空/失败 | 夜间重启服务后累计数据恢复；07:00 正常生成完整夜图 | 待修 |
| **2** | 告警帧 best-effort：队列满丢、`/events` 仅 200 条、服务不持久化；单线程 HTTPServer 阻塞也会丢 | 告警数据不可可靠追溯 | 压测队列满、HTTP 阻塞、服务重启等场景，告警不静默丢失 | 待修 |
| **3** | Linux 宿主无法解析 `host.docker.internal` | 回调实际全部失败 | ~~Linux Docker 环境实测 callback 成功~~ | **忽略**（见下） |
| **8** | Reader 断流只静默重连，没有断流告警 | 整晚掉线可能无人知晓 | 模拟持续断流，平台收到明确 degraded/断流事件；恢复后有恢复状态 | 待修 |

### #3 降级说明

最终回调路径是**网络 webhook URL**，不使用 `host.docker.internal`；该值会随部署环境替换。
因此**不做 Linux 宿主可达性测试与 `extra_hosts` 适配**，从 Blocker 移出并忽略。

---

## 二、P1 — 上线前最好一起收掉

### Payload / 数据契约

- **P-a**：Alarm payload 信息不足。
  缺 `box / rer / severity / row_id / lamp_id / kind / algo`，下游无法回答“哪一行/哪盏灯/什么类型/置信度/算法判断”。
  处理：把 Alarm payload 定为**唯一正式契约**，禁止第二套 legacy payload。
- **P-b**：Report payload 缺 `status(ok|degraded|failed)` 与 `metadata`。
  后果：平台无法区分正常/降级/失败夜图，放大 #4、#8。

### 队列 / 节流

- **Q-a**：`submit_event` 顺序有 bug——先写 throttle 再入队；队列满丢弃时 throttle 已记账，导致 3600s 不再发送。
  处理：改为**入队/接收成功后再更新 throttle**，或严格定义 “accepted” 语义。

### Report / 接收端

- **R-a**：接收端摘要读 `objects`，现行格式恒为 0；且仍是单线程。
  处理：**拆成两个 issue**——(1) 摘要字段与现行 payload 对齐；(2) 接收端并发模型（`ThreadingHTTPServer`）。

### 生命周期

- **C-d**：停机不干净，`while True` worker 不退出、不 `join`。
  与 #1 直接叠加（重启 → 无 graceful shutdown → 状态未 flush → 夜间累计丢失）。
  **建议与 #1 一起修。**

### 状态 / 报告正确性

- **#4**：整夜 `status=degraded`（实测 `n_cand_total=426 > max_candidates=200`）。
  注意：**不是算法逻辑问题**，是阈值/状态判定把一个实际可运行的整夜结果标成 degraded。
  处理：调大 `max_candidates` 和/或明确平台侧 `degraded` 语义，否则“正常产图 → degraded → 平台整晚报警”。
- **#11a**：`python:3.12-slim` 无 tzdata，回退固定 `+8`；`RealClock / Reporter` 时间是否落 UTC 需容器内实测。
  处理：**先实测不直接改码**，核对链路 `container tz → RealClock.now() → Reporter timestamp → report_at → day_start`。

---

## 三、P2 — 集中 cleanup

| Issue | 内容 | 性质 |
| --- | --- | --- |
| C-a | `report_at == day_start` 靠边界排序 | 状态机边界风险 |
| C-b | evidence 特性 + 类不存在；`notify()` legacy | 死代码 / 第二契约 |
| C-c | loguru 未配置 | 日志配置失效 |
| C-e | `output_dir` 相对路径依赖 CWD | 部署鲁棒性 |
| P-c | `image_base64` / `frame_base64` 两套默认值 | payload 契约不统一 |
| P-d | Reporter 与 notify 两套 payload 并存 | 契约不统一 |
| R-b | `/events` 没增量游标 | 查询效率 / 可用性 |
| 9 | `memory_budget_mb_per_camera` 只是 manifest-only | 命名误导 |
| 10 | key 手写集合、`NIGHT_GATE_KEYS`、`night_gate` 裸 dict | 配置结构可维护性 |

---

## 四、业务决策项（单独隔离）

### #6 alarm vs suspected

现状：只推 `alarm` + `alarm_min_interval_s = 3600` + 按 row 节流
→ 同一行一小时内第二次真实缺口不再推 alarm。

**不得当 bug 直接修掉（业务策略）。** 需业务明确：

- 方案 A：继续只推 alarm，`interval = 3600s`
- 方案 B：alarm 仍 3600s，`suspected` 单独推送
- 方案 C：放宽 `alarm_min_interval_s`

标记：**Decision Required**，不并入工程 TODO。

---

## 五、明确排除项（置于顶部防 scope creep）

### 不做

- **#7 鉴权 / TLS**：新需求，不属于当前缺陷修复范围。
  当前保持 `0.0.0.0:8000` 无鉴权、HTTP callback。

### 撤销

- **#5**：水马漏报属场景/算法逻辑，本轮不处理。
- **#11b**：`opencv_python-5.0.0.93` 与 pin wheel 一致，`5.0.0` vs `5.0.0.93` 仅为 `__version__` 展示差异，非版本不一致。
- **#4 数字**：`432 → 426`，结论不变。

---

## 六、执行顺序（按数据依赖，不按 P 级）

1. **第一组 — 数据生命周期**：`#1 夜间持久化` → `C-d graceful shutdown` → `#2 告警可靠落盘/队列`
   解决：数据到底能不能活下来。
2. **第二组 — 告警链路**：`#3（已忽略）`、`#8 断流告警` → `Q-a throttle 顺序` → `P-a Alarm payload` → `P-b Report payload`
   解决：数据活下来后能不能正确通知出去。
3. **第三组 — 状态/报告正确性**：`#4 degraded` → `R-a objects` → `#11a timezone`
   解决：平台收到的数据是否正确描述实际运行状态。
4. **第四组 — cleanup**：`C-a / C-b / C-c / C-e`、`P-c / P-d`、`R-b`、`#9 / #10`

---

## 七、Release Checklist

```text
BarriGuard Release Gate

[P0 — BLOCKER]
[x] #1  夜间累计持久化
[x] #2  告警帧可靠性（服务端落盘）
[~] #3  Linux callback host            （忽略：最终走网络 webhook URL）
[x] #8  Reader 断流告警

[P1 — REQUIRED]
[x] P-a  Alarm payload 完整化
[x] P-b  Report payload 完整化
[x] Q-a  throttle 入队顺序（随 #2 一并修）
[ ] R-a  objects 摘要 + 接收端并发模型（objects 已随 P-a 补齐；并发待做）
[x] C-d  graceful shutdown（与 #1 同组）
[ ] #4   degraded 判定
[ ] #11a timezone 实测/修复

[P2 — CLEANUP]
[ ] C-a
[ ] C-b
[ ] C-c
[ ] C-e
[ ] P-c
[ ] P-d
[ ] R-b
[ ] #9
[ ] #10

[DECISION REQUIRED]
[ ] #6   alarm / suspected / throttle 策略

[OUT OF SCOPE]
[ ] #7   Auth / TLS

[RETRACTED]
[x] #5
[x] #11b
[x] #4 数字修正：426
```

---

## 八、第一组完成记录（#1 + C-d + #2）

实现：

- `night_lamp/persist.py`：`NightStateStore` 原子 npz 快照（临时文件 + `os.replace`），`night_key` 以 12:00 为界归夜。
- `night_lamp/session.py`：`dump_state()` / `load_state()`（空间累计 `_base/_bg/_on/_peak` + `_n_seen` + gate），新增 `camera_id` / `restored`；报告 metadata 加 `restored`。
- `night_lamp/adapter.py`：`attach_persistence()` / `restore()` / `save_state()`；burst 边界、freeze、连续模式周期（600s）落盘；finalize 后清理。
- `server/worker.py`：新增 `night_store` 注入与 `start_night` 恢复、`flush_night()`；`_loop` 可停、`stop()` join；修 fps 日志除零。
- `server/schedule.py`：启动于 `FROZEN`（03:00–07:00）补发 `NIGHT_START + NIGHT_FREEZE`，保 07:00 出图。
- `server/source.py`：`Reader.stop()` join 采集线程。
- `server/alarms.py`：`AlarmStore` 告警帧（jpg + json）服务端落盘。
- `server/reporting.py`：`submit_event` 先落盘再入队，节流据「落盘成功」记账（Q-a）。

验收：`deploy/tests/acceptance/test_reliability.py`，全套 24/24 PASS。

遗留（不在本组）：#2 的 callback 端仍 best-effort（无重启后重放）；`/events` 内存 200 条与接收端并发归 R-a/R-b；`#8 断流告警` 为第二组。

---

## 九、第二组完成记录（#8 + P-a + P-b）

实现：

- `server/source.py`：`Reader.health()`（connected / last_frame_mono / started_mono / reconnects），拉流成功置连、断流置断并计数。
- `server/worker.py`：`_check_stream_health()`（无帧超 `STREAM_LOST_AFTER_S=10s` 推 `stream_lost`，恢复推 `stream_restored`，持续断流每 `STREAM_HEARTBEAT_S=600s` 心跳），状态写入 `/events` 的 `status`。
- `server/reporting.py`：新增 `submit_status()`（`alert_type=camera_status`，不参与 alarm 节流）；`_body` 补业务字段——event 带 `kind/algo/target/bbox/objects/metadata`，report 带 `report_type/algorithm/status/metadata`；`_send` 支持预编码 `raw`。

验收：`test_reliability` 新增 `stream_health_alert`、`payload_contract`，全套 26/26 PASS。

遗留（下一组）：R-a 接收端并发；#4 degraded 判定；#11a timezone。


