# rag_engine.py — 混合检索（Chroma 向量 + BM25）
import app.env_setup  # noqa: F401
from pathlib import Path
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge.txt"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

_embeddings = None
_vector_retriever = None
_bm25_retriever = None


def chunk_documents(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Document]:
    """按固定长度切分文本，避免依赖 langchain_text_splitters（部分环境 SSL 导入失败）。"""
    chunks: list[Document] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Document(page_content=text[start:end], metadata={"source": source}))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _ensure_chroma() -> Path:
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            "未找到向量库 chroma_db。请先运行：\n"
            "  python scripts/build_vectordb.py"
        )
    return CHROMA_DIR


def _get_retrievers():
    global _vector_retriever, _bm25_retriever
    if _vector_retriever is None:
        chroma_dir = _ensure_chroma()
        vectordb = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=_get_embeddings(),
        )
        _vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

        if not KNOWLEDGE_FILE.exists():
            raise FileNotFoundError(f"知识库文件不存在：{KNOWLEDGE_FILE}")

        text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
        chunks = chunk_documents(text, str(KNOWLEDGE_FILE))
        _bm25_retriever = BM25Retriever.from_documents(chunks)
        _bm25_retriever.k = 10
    return _vector_retriever, _bm25_retriever


def hybrid_retrieve(question: str, top_k: int = 3) -> list[Document]:
    """向量 + BM25 融合检索，返回 top_k 文档片段。"""
    vector_retriever, bm25_retriever = _get_retrievers()
    docs_vector = vector_retriever.invoke(question)
    docs_bm25 = bm25_retriever.invoke(question)

    score_dict: dict[str, float] = defaultdict(float)
    content_to_doc: dict[str, Document] = {}

    for doc in docs_vector:
        score_dict[doc.page_content] += 1.0
        content_to_doc[doc.page_content] = doc
    for doc in docs_bm25:
        score_dict[doc.page_content] += 1.0
        content_to_doc[doc.page_content] = doc

    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [content_to_doc[content] for content, _ in sorted_items]
