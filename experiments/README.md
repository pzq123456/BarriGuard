# experiments/

水马识别试验代码。输出一律写入 `results/<时间戳>_<实验名>/`，每轮独立，不覆盖。

## 主流程

| 脚本 | 用途 | 状态 |
|---|---|---|
| `barrier_bin.py` | 水马语义二值化（红∪白 × 近严远宽阈值 × ROI × 定向闭运算 → 0/255 mask） | 现行方案，业务目标 |
| `barrier_seg.py` | 水马实例分割（红/白掩膜 → watershed 两遍 → 堆叠合并） | 旧路线，远端漏检，留档 |
| `common.py` | 公共工具：输入加载、run 目录、裁剪 | 被上面引用 |

## 诊断（调阈值时用，非主流程）

| 脚本 | 用途 |
|---|---|
| `diagnostics/sample_colors.py` | 打印采样点 HSV/Lab 表，标定阈值用 |
| `diagnostics/red_cc.py` | 红色连通域特征表 + 可视化，调 RED_* 阈值 |
| `diagnostics/white_cc.py` | 白色连通域特征表，调 WHITE_* 阈值 |

## 目录约定

- `data/input/`：原始帧 `<相机>_<时分>.png`，只进不改
- `results/`：试验输出，每轮一个子目录；`_legacy_tmp_out/` 是旧 tmp/out 的归档，可删
