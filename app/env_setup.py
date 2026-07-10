# env_setup.py — 必须在 huggingface 相关 import 之前加载
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_BASE_URL", "https://hf-mirror.com")
