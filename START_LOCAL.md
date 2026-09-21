# BarriGuard 本机最小启动指南

给执行者（人或 AI）：按顺序复制执行即可。**所有命令在仓库根目录运行**。
仓库根：`C:\Users\admin\Desktop\work\BarriGuard`

## 前置检查（一次性）

```powershell
$root = "C:\Users\admin\Desktop\work\BarriGuard"
Test-Path "$root\.venv\Scripts\python.exe"          # 必须 True
Get-NetTCPConnection -State Listen -EA SilentlyContinue | ? { $_.LocalPort -in 8000,998 }
```
上面最后一条应**无输出**；若有输出，先停掉占用 8000/998 的进程。

## 启动

```powershell
$root = "C:\Users\admin\Desktop\work\BarriGuard"
$py   = "$root\.venv\Scripts\python.exe"

# 1) 消费端（接收回调，落盘到 output\alerts）
Start-Process $py -ArgumentList 'local\webhook_receiver.py --host 127.0.0.1 --port 998 --save-dir output\alerts' -WorkingDirectory $root

# 2) 检测服务（本机配置；回调 127.0.0.1:998）
$env:PYTHONPATH = "$root\deploy\app"
$env:BARRIGUARD_CONFIG = "$root\deploy\config.local.yaml"
Start-Process $py -ArgumentList '-m server' -WorkingDirectory $root
```

两个进程都后台运行，可关闭当前窗口。

## 验证（启动后 30 秒）

```powershell
Invoke-WebRequest http://127.0.0.1:8000/cameras -UseBasicParsing | % Content
Invoke-WebRequest http://127.0.0.1:8000/snapshot/1749.jpg -UseBasicParsing | % RawContentLength
Get-NetTCPConnection -State Listen -EA SilentlyContinue | ? { $_.LocalPort -in 8000,998 } | ft LocalPort
```
期望：`/cameras` 返回 1749/1750；`snapshot` 大小 > 100000；8000 与 998 均在监听。

## 结果落盘位置

| 内容 | 路径 |
|---|---|
| 消费端收到的帧+payload | `output\alerts\<camera>\*.jpg|json`、`output\alerts\payload.jsonl` |
| 服务端报告（日图/夜图） | `data\<camera>\*.jpg|json` |
| 服务端告警帧 | `data\alerts\<camera>\*.jpg|json` |
| 夜灯核心数据（L1 归档） | `data\night_archive\<夜>\<camera>\state.npz` + `manifest.json` |

## 时间预期（时区 Asia/Shanghai）

- 07:00–17:00 白昼：水马每小时出图。
- 20:00–03:00 夜间：每小时出 burst 热力图；**07:00 出整夜最终热力图**。
- 夜图可能 `status=degraded`（候选溢出，属已知项）。

## 停止

```powershell
foreach ($p in 8000,998) {
  Get-NetTCPConnection -State Listen -LocalPort $p -EA SilentlyContinue |
    % { Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue }
}
```

## 注意

- 必须从仓库根启动（`output_dir`/`data` 是相对路径）。
- 启动后**不要在 07:00 前停服务**，否则拿不到整夜热力图。
- 机器不要休眠，保持 RTSP 可达。
- 若要改回调地址/出图节奏，编辑 `deploy\config.local.yaml`（生产 `config.yaml` 不动）。
