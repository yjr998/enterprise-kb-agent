# build_vectordb.py — 一键构建 Chroma 向量库
"""用法：python scripts/build_vectordb.py"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app.env_setup  # noqa: F401

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.rag_engine import KNOWLEDGE_FILE, CHROMA_DIR, chunk_documents


def main():
    if not KNOWLEDGE_FILE.exists():
        print(f"错误：找不到 {KNOWLEDGE_FILE}")
        sys.exit(1)

    print("1/4 加载文档...")
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    print("2/4 切分文档...")
    chunks = chunk_documents(text, str(KNOWLEDGE_FILE))
    print(f"    共 {len(chunks)} 个片段")

    print("3/4 加载 embedding 模型（首次约 33MB）...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("4/4 写入 Chroma...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    count = vectordb._collection.count()
    print(f"完成：{count} 条向量已写入 {CHROMA_DIR}")


if __name__ == "__main__":
    main()
