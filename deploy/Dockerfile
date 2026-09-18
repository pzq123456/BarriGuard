# BarriGuard 运行时镜像 —— 纯 Python，无 apt、无系统 ffmpeg。
#
# 为什么不需要 ffmpeg 二进制：
#   server/source.py 在 shutil.which("ffmpeg") 为空时回退到
#   cv2.VideoCapture(CAP_FFMPEG)，用的是 opencv-python(-headless) wheel 自带的
#   FFmpeg。已在真实 1749 的 HEVC 流上实测（1920x1080@25，60 帧无灰帧）。
#   用 headless 变体还免掉了 libGL / libSM / libXext 等系统库。
#
# 构建上下文 = deploy/ 目录（见 docker-compose.yml），自包含生产世界：
#   app/ 下 server/ water_barrier/ night_lamp/ 三个包，包名不变、import 不变；
#   COPY app/ /app/ 后 PYTHONPATH=/app，`python -m server` 直接可用。
# 标定 YAML 与 registry CSV 就在两个算法包里，随 app/ 一起进镜像。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BARRIGUARD_CONFIG=/app/config.yaml \
    TZ=Asia/Shanghai

WORKDIR /app

# 版本对齐 uv.lock。
# 不装 torch / torchvision / ultralytics / scipy / simplejpeg：源码零 import。
RUN pip install --no-cache-dir \
    "fastapi==0.141.1" \
    "uvicorn==0.52.4" \
    "loguru==0.7.3" \
    "numpy==2.5.2" \
    "opencv-python-headless==5.0.0.93" \
    "pyyaml==6.0.3"

# 自包含生产世界：上下文内 app/ 即容器内 /app（包名不变）。
COPY app/ /app/
# 内置一份配置作为兜底；运行期由 docker-compose 挂载覆盖（见 compose）。
COPY config.yaml /app/config.yaml

# 可写目录：
#   evidence/           报告/证据落盘兜底目录
#   night_lamp/output/  night_lamp config 里 evidence.out 相对输出目录
RUN mkdir -p /app/evidence /app/night_lamp/output

EXPOSE 8000

# /cameras 很轻，且能证明 config 已加载、app 已成功 import。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/cameras', timeout=4)"

# 入口只有这一个：python -m server 会 uvicorn.run("server.app:app")。
# server/config.py 优先读 env BARRIGUARD_CONFIG（=/app/config.yaml）。
CMD ["python", "-m", "server"]
