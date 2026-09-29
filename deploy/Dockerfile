# BarriGuard 运行时镜像 —— python:3.12-slim + 一个静态 ffmpeg 二进制。
#
# 为什么必须自带 ffmpeg（2026-09-29 实测）：
#   1749 是 HEVC 流。cv2(opencv-python-headless) 自带 FFmpeg、imageio-ffmpeg 的
#   essentials 构建、以及 Debian 的 ffmpeg 7.1 解这条流都会**持续输出灰帧**
#   （luma std≈2~7；正常画面 40~70），导致水马白天一直 OCCLUDED、零输出。
#   只有较新的 ffmpeg（BtbN master, 2026）能正常解码，故用多阶段构建把它的
#   ffmpeg 二进制放进最终镜像的 PATH；最终镜像仍是 slim，只多一个二进制文件。
#
# 构建上下文 = deploy/ 目录（见 docker-compose.yml）：
#   app/ 下 server/ water_barrier/ night_lamp/ 三个包，包名不变、import 不变；
#   COPY app/ /app/ 后 PYTHONPATH=/app，`python -m server` 直接可用。
#   水马标定 YAML 在各算法包里，随 app/ 一起进镜像。

FROM debian:bookworm-slim AS ffmpeg
# 需要 ffmpeg>=8/master；Debian 7.1 与 essentials 构建均会灰帧。
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "$FFMPEG_URL" -o /tmp/ff.tar.xz \
 && tar -xf /tmp/ff.tar.xz -C /tmp \
 && cp /tmp/ffmpeg-*/bin/ffmpeg /opt/ffmpeg \
 && /opt/ffmpeg -version | head -1

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BARRIGUARD_CONFIG=/app/config.yaml \
    TZ=Asia/Shanghai

WORKDIR /app

# 版本对齐 uv.lock。不装 torch / torchvision / ultralytics / scipy / simplejpeg。
RUN pip install --no-cache-dir \
    "fastapi==0.141.1" \
    "uvicorn==0.52.4" \
    "loguru==0.7.3" \
    "numpy==2.5.2" \
    "opencv-python-headless==5.0.0.93" \
    "pyyaml==6.0.3" \
    "tzdata==2026.4"

# 静态 ffmpeg（见文件头）。放 PATH，source.py 的 shutil.which("ffmpeg") 直接命中。
COPY --from=ffmpeg /opt/ffmpeg /usr/local/bin/ffmpeg

# 自包含生产世界：上下文内 app/ 即容器内 /app（包名不变）。
COPY app/ /app/
# 内置一份配置作为兜底；运行期由 docker-compose 挂载覆盖（见 compose）。
COPY config.yaml /app/config.yaml

# 可写目录：
#   evidence/           报告/证据落盘兜底目录
#   night_lamp/output/  night_lamp config 里 evidence.out 相对输出目录
RUN mkdir -p /app/evidence /app/night_lamp/output /app/data

EXPOSE 8000

# /cameras 很轻，且能证明 config 已加载、app 已成功 import。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/cameras', timeout=4)"

# 入口只有这一个：python -m server 会 uvicorn.run("server.app:app")。
# server/config.py 优先读 env BARRIGUARD_CONFIG（=/app/config.yaml）。
CMD ["python", "-m", "server"]
