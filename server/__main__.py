"""运行入口。

  python -m server                        # 启动实时HTTP预览（按 server/config.yaml）
  python -m server --config <path>        # 指定配置文件
  python -m server --offscreen 图          # 离线跑N遍同一帧，验证时序收敛到ALARM
"""
import argparse
import os
import sys

import cv2 as cv
from loguru import logger

from .config import load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def offscreen(params, img_path: str, repeat: int = 14,
              outdir: str = "water_barrier/research/out") -> None:
    """不依赖 HTTP 的端到端校验：对同一帧重复喂给引擎，观察状态机 NORMAL->SUSPECTED->ALARM。"""
    from server.engine import Monitor
    frame = cv.imread(img_path)
    mon = Monitor(frame.shape, params.water_gap)
    os.makedirs(outdir, exist_ok=True)
    for i in range(repeat):
        views = mon.step(frame, i * 1.0)  # 每帧间隔1s，加速累计
        if i % (repeat // 4 or 1) == 0 or i == repeat - 1:
            logger.info("t={}s  轨道={}  告警={}", i, len(views),
                        [(v.kind, v.state.name, round(v.rer, 2)) for v in views])
    vis = frame.copy()
    for v in mon.step(frame, repeat):
        x0, y0, x1, y1 = v.box
        cv.rectangle(vis, (x0, y0), (x1, y1),
                     (80, 80, 255) if v.state.name == "ALARM" else (80, 160, 255), 2)
        cv.putText(vis, f"{v.kind}:{v.state.name}", (x0, max(20, y0 - 6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out = os.path.join(outdir, os.path.splitext(os.path.basename(img_path))[0] + "_state.png")
    cv.imwrite(out, vis)
    logger.info("状态叠加已保存: {}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="配置文件路径（默认 server/config.yaml）")
    ap.add_argument("--offscreen", metavar="IMG", help="离线验证模式(喂同一帧N遍)")
    ap.add_argument("--repeat", type=int, default=14)
    args = ap.parse_args()

    params = load(args.config)
    if args.offscreen:
        offscreen(params, args.offscreen, args.repeat)
        return

    import uvicorn
    p = params.server
    logger.info("BarriGuard 启动: http://{}:{}", p.host, p.port)
    uvicorn.run("server.app:app", host=p.host, port=p.port, log_level="warning")


if __name__ == "__main__":
    main()
