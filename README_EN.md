# Enterprise Knowledge Base Agent

An on-premise enterprise Q&A agent powered by **RAG + LangGraph**, with hybrid retrieval, multi-tool orchestration, and multi-turn dialogue via FastAPI.

[中文 README](README.md)

---

## Features

- **Hybrid Retrieval**: Chroma vector search + BM25 keyword search
- **LangGraph Agent**: Routes to knowledge search, order lookup, or calculator tools
- **Multi-turn Chat**: Session-based conversation history
- **One-click Start**: `start.bat` on Windows
- **Evaluation**: 20 test cases with `run_eval.py`

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-kb-agent.git
cd enterprise-kb-agent

pip install -r requirements.txt
ollama pull qwen2:1.5b

python scripts/build_vectordb.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

---

## Architecture

```
User → FastAPI → LangGraph Agent → Tools (RAG / Order / Calculator) → Ollama
```

Hybrid retrieval combines semantic (Chroma) and lexical (BM25) search before LLM generation.

---

## Tech Stack

Python · FastAPI · LangGraph · Chroma · Ollama · bge-small-zh

---

## License

MIT
