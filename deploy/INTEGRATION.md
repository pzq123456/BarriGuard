# BarriGuard 接入说明

账号需求

```
項目就是古洞C2 1750和1749兩個攝像頭

账号名是new AI demo
密码是esggesg2026
我們那兩個攝像頭（包括AI功能）只能這個賬號看到
```

## 1. 回调地址

一个 **HTTP 接口地址**（“回调地址”），能接收 `POST`、`Content-Type: application/json` 的 JSON。

## 2. 字段

每条消息公共字段：

| 字段 | 说明 |
|---|---|
| `alert_type` | 消息类型：`water_gap`｜`night_heatmap` |
| `camera_id` / `camera_name` | 相机号 / 名称 |
| `timestamp` | 时间（UTC） |
| `frame_base64` | 图片（JPEG 的 base64），无图时为 `""` |

**① 水马报警（白天检测到缺口）**
```json
{
  "alert_type": "water_gap",
  "camera_id": "1750",
  "camera_name": "Mobile Camera 1750",
  "timestamp": "2026-09-22T00:45:49.808415+00:00",
  "frame_base64": "/9j/4AAQ...",
  "kind": "alarm",
  "target": "row_0_left_near",
  "bbox": [170, 239, 269, 401],
  "metadata": {"rer": 0.848, "severity": 0.928, "row_id": "row_0_left_near"}
}
```

**② 夜间热力图（每小时一张 + 早上整夜一张）**
```json
{
  "alert_type": "night_heatmap",
  "camera_id": "1749",
  "camera_name": "Mobile Camera 1749",
  "timestamp": "2026-09-22T05:39:00+00:00",
  "frame_base64": "/9j/4AAQ...",
  "report_type": "night_heatmap_burst",
  "status": "ok",
  "metadata": {"bucket": "2026-09-21T22", "coverage": 0.95, "silent_bucket": false, "n_flash": 68}
}
```
- `report_type`：`night_heatmap_burst`=逐小时图；`night_heatmap`=早上整夜图。
- `silent_bucket=true` 表示该小时没采到数据（断流），仍会发一张。

## 3. 什么时候发

| 时间（北京） | 发送内容 |
|---|---|
| 白天 07:00–17:00 | **检测到水马缺口就发**（同一处有冷却，不会刷屏） |
| 夜间 20:00–03:00 | **每小时一张热力图**：20:10、21:10、…、02:10 |
| 早上 07:00 | 整夜累计热力图一张 |

> 图片解码：把 `frame_base64` 做 base64 解码即为 `.jpg`。
> 时间戳是 UTC，展示时 +8 小时为北京时间。
