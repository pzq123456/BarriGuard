"""python -m server：启动拉流 + 算法 + HTTP 服务。"""
import sys

import uvicorn
from loguru import logger

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOST, PORT = "127.0.0.1", 8000


def main():
    logger.info("BarriGuard 启动: http://{}:{}", HOST, PORT)
    uvicorn.run("server.app:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
