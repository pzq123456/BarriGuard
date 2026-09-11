"""运行入口。

  python -m server                        # 启动实时HTTP预览（按 server/config.yaml）
  python -m server --config <path>        # 指定配置文件
  python -m server --offscreen 图          # 离线跑N遍同一帧，验证时序收敛到ALARM
  python -m server --offscreen 图 --camera 1750  # 指定相机的标定跑离线验证
"""
import argparse
import os
import sys

import cv2 as cv
from loguru import logger

from . import registry, render
from .config import load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def offscreen(params, img_path: str, repeat: int = 14, camera: str = None,
               outdir: str = "water_barrier/output",
               algo: str = "water_gap") -> None:
    """不依赖 HTTP 的端到端校验：对同一帧重复喂给算法，观察事件收敛。

    只适用于单帧算法（同一帧喂 N 遍）；时序算法（night_lamp，靠连续
    不同帧 + 去重）用静态帧验不出东西，明确拒绝，不静默跑错。
    """
    cams = [c for c in params["cameras"] if c["algos"]]
    if camera:
        cams = [c for c in cams if c["id"] == camera]
    if not cams:
        raise RuntimeError("所选相机没有启用的算法")
    cam = cams[0]
    if algo not in cam["algos"]:
        raise RuntimeError("相机 %s 未启用算法 %s（可用：%s）"
                           % (cam["id"], algo, sorted(cam["algos"])))
    if algo != "water_gap":
        raise RuntimeError("%s 是时序算法，offscreen 静态帧模式不适用" % algo)
    name = algo
    frame = cv.imread(img_path)
    algo = registry.create(name, frame.shape, cam["algos"][name], cam["id"])
    os.makedirs(outdir, exist_ok=True)
    for i in range(repeat):
        res = algo.step(frame, i * 1.0)  # 每帧间隔1s，加速累计
        if i % (repeat // 4 or 1) == 0 or i == repeat - 1:
            logger.info("[{}:{}] t={}s 标注={} 事件={}", cam["id"], name, i,
                        len(res.annots),
                        [(e.kind, e.payload.get("rer")) for e in res.events])
    res = algo.step(frame, repeat)
    vis = render.draw_status(render.draw_annots(frame.copy(), res.annots),
                             res.debug.get("frame_status", "OK"))
    out = os.path.join(outdir, os.path.splitext(os.path.basename(img_path))[0] + "_state.png")
    cv.imwrite(out, vis)
    logger.info("状态叠加已保存: {}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="配置文件路径（默认 server/config.yaml）")
    ap.add_argument("--offscreen", metavar="IMG", help="离线验证模式(喂同一帧N遍)")
    ap.add_argument("--repeat", type=int, default=14)
    ap.add_argument("--camera", help="离线验证用的相机 id（默认首个有算法的相机）")
    ap.add_argument("--algo", default="water_gap", help="离线验证用的算法（默认 water_gap）")
    args = ap.parse_args()

    params = load(args.config)
    if args.offscreen:
        offscreen(params, args.offscreen, args.repeat, args.camera,
                  algo=args.algo)
        return

    import uvicorn
    p = params["server"]
    logger.info("BarriGuard 启动: http://{}:{}", p["host"], p["port"])
    uvicorn.run("server.app:app", host=p["host"], port=p["port"], log_level="warning")


if __name__ == "__main__":
    main()
