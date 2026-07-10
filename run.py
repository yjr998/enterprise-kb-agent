# run.py — 启动 Web 服务（PowerShell / CMD 通用）
"""用法：python run.py"""
import app.env_setup  # noqa: F401 — 设置 HF 镜像

import uvicorn

if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in browser")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
