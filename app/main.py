# main.py — FastAPI Web 服务
import app.env_setup  # noqa: F401
import uuid

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage, AIMessage

from app.agent import chat, new_session_messages

app = FastAPI(title="Enterprise KB Agent", version="1.0.0")

_sessions: dict[str, list] = {}

HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>企业知识库 Agent</title>
  <style>
    body {{ font-family: sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ color: #222; }}
    .hint {{ color: #666; font-size: 14px; }}
    textarea {{ width: 100%; height: 72px; font-size: 16px; padding: 8px; }}
    button {{ margin-top: 10px; padding: 10px 24px; font-size: 16px; cursor: pointer; margin-right: 8px; }}
    .history {{ margin-top: 24px; }}
    .msg {{ padding: 12px; margin: 8px 0; border-radius: 8px; }}
    .user {{ background: #e3f2fd; }}
    .bot {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>企业知识库 Agent</h1>
  <p class="hint">RAG 检索 · 订单查询 · 计算器 · 多轮对话</p>
  <form action="/chat" method="post">
    <input type="hidden" name="session_id" value="{session_id}">
    <textarea name="question" placeholder="例如：人工智能会取代哪些工作？ / 订单12345能退吗？" required></textarea>
    <br>
    <button type="submit">发送</button>
    <button type="submit" formaction="/reset" formmethod="post">新对话</button>
  </form>
  <div class="history">{history_block}</div>
</body>
</html>
"""


def _get_session(session_id: str | None) -> tuple[str, list]:
    if not session_id or session_id not in _sessions:
        sid = str(uuid.uuid4())[:8]
        _sessions[sid] = new_session_messages()
        return sid, _sessions[sid]
    return session_id, _sessions[session_id]


def _render_history(session_id: str) -> str:
    msgs = _sessions.get(session_id, [])
    blocks = []
    for m in msgs[1:]:
        content = getattr(m, "content", "")
        if isinstance(m, HumanMessage) and content:
            blocks.append(f'<div class="msg user"><b>你：</b>{content}</div>')
        elif isinstance(m, AIMessage) and content and not getattr(m, "tool_calls", None):
            blocks.append(f'<div class="msg bot"><b>Agent：</b>{content}</div>')
    return "\n".join(blocks) if blocks else "<p>暂无对话，请输入问题。</p>"


@app.get("/", response_class=HTMLResponse)
def home():
    sid, _ = _get_session(None)
    return HTML_PAGE.format(session_id=sid, history_block="")


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/chat", response_class=HTMLResponse)
def chat_form(session_id: str = Form(""), question: str = Form(...)):
    sid, messages = _get_session(session_id or None)
    updated, _ = chat(messages, question.strip())
    _sessions[sid] = updated
    return HTML_PAGE.format(session_id=sid, history_block=_render_history(sid))


@app.post("/reset", response_class=HTMLResponse)
def reset(session_id: str = Form("")):
    if session_id in _sessions:
        del _sessions[session_id]
    sid, _ = _get_session(None)
    return HTML_PAGE.format(session_id=sid, history_block="")


@app.post("/api/chat")
def chat_api(session_id: str = Form(""), question: str = Form(...)):
    sid, messages = _get_session(session_id or None)
    updated, answer = chat(messages, question.strip())
    _sessions[sid] = updated
    return JSONResponse({"session_id": sid, "question": question, "answer": answer})
