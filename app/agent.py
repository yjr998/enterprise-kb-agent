# agent.py — LangGraph 多工具 Agent
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from app.rag_engine import hybrid_retrieve

MOCK_ORDERS = {
    "12345": {"product": "无线蓝牙耳机", "status": "已签收", "days_since_buy": 3},
    "67890": {"product": "机械键盘", "status": "运输中", "days_since_buy": 1},
    "11111": {"product": "显示器支架", "status": "已签收", "days_since_buy": 15},
}

RETURN_POLICY = (
    "退货政策：签收后7天内可无理由退货；退款3-5工作日到账；"
    "客服热线 400-123-4567。"
)


@tool
def search_docs(query: str) -> str:
    """搜索企业知识库。用户问 AI、产品、政策、新闻等需查资料的问题时使用。"""
    try:
        docs = hybrid_retrieve(query, top_k=3)
        if not docs:
            return "知识库中未找到相关信息。"
        return "\n\n".join(f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs, 1))
    except Exception as e:
        return f"检索失败：{e}"


@tool
def lookup_order(order_id: str) -> str:
    """查询订单状态。用户提供5位订单号时使用。"""
    order_id = re.sub(r"\D", "", order_id)[-5:] if order_id else ""
    if not order_id:
        return "请提供5位订单号。"
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}。"
    return (
        f"订单号：{order_id}，商品：{order['product']}，"
        f"状态：{order['status']}，购买{order['days_since_buy']}天。{RETURN_POLICY}"
    )


@tool
def add(a: float, b: float) -> float:
    """两数相加。"""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """两数相减。"""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """两数相乘。"""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """两数相除。"""
    if b == 0:
        return float("nan")
    return a / b


tools = [search_docs, lookup_order, add, subtract, multiply, divide]

# qwen2:1.5b 不支持 Ollama tool calling，需用 qwen2.5:3b（或支持 tools 的模型）
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content="""你是企业知识库智能助手，可使用工具：

- search_docs：查知识库（AI、工作、产品信息等）
- lookup_order：查订单（需5位订单号）
- add/subtract/multiply/divide：数学计算

规则：
1. 需要事实依据 → 必须 search_docs，不要编造
2. 有订单号或问物流 → lookup_order
3. 数学计算 → 调用计算器，不要心算
4. 中文礼貌回答，简洁专业""")


def chatbot(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

agent_app = graph_builder.compile()


def new_session_messages():
    return [SYSTEM_PROMPT]


def _extract_answer(messages: list) -> str:
    """取最后一条带内容的 AI 回复（跳过 tool_call 中间态）。"""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = (msg.content or "").strip()
        if content:
            return content
    return ""


def _try_quick_answer(user_input: str) -> str | None:
    """规则兜底：小模型偶发不调工具时，保证基础能力可用。"""
    text = user_input.strip()

    order_match = re.search(r"\d{5}", text)
    if order_match and any(k in text for k in ("订单", "退货", "物流", "签收", "买了")):
        return lookup_order.invoke({"order_id": order_match.group(0)})

    math_match = re.search(
        r"(\d+(?:\.\d+)?)\s*([+\-*/×x除以加减乘])\s*(\d+(?:\.\d+)?)", text
    )
    if math_match:
        a, op, b = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
        ops = {
            "+": add, "加": add,
            "-": subtract, "减": subtract,
            "*": multiply, "x": multiply, "×": multiply, "乘": multiply,
            "/": divide, "除以": divide,
        }
        fn = ops.get(op)
        if fn:
            result = fn.invoke({"a": a, "b": b})
            if result != result:  # NaN
                return "错误：除数不能为 0。"
            op_display = {"+": "+", "加": "+", "-": "-", "减": "-", "*": "×", "x": "×", "×": "×", "乘": "×", "/": "÷", "除以": "÷"}.get(op, op)
            return f"{int(a) if a == int(a) else a} {op_display} {int(b) if b == int(b) else b} = {int(result) if result == int(result) else result}。"

    return None


def chat(session_messages: list, user_input: str) -> tuple[list, str]:
    from langchain_core.messages import AIMessage

    quick = _try_quick_answer(user_input)
    if quick:
        session_messages.append(HumanMessage(content=user_input))
        session_messages.append(AIMessage(content=quick))
        return session_messages, quick

    session_messages.append(HumanMessage(content=user_input))
    result = agent_app.invoke({"messages": session_messages}, config={"recursion_limit": 12})
    updated = result["messages"]
    return updated, _extract_answer(updated)
