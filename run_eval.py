# run_eval.py — 批量评测 Agent
"""用法：python run_eval.py"""
import json
import time
from pathlib import Path

import app.env_setup  # noqa: F401
from app.agent import chat, new_session_messages

CASES_FILE = Path(__file__).parent / "data" / "test_cases.json"
OUT_FILE = Path(__file__).parent / "docs" / "evaluation_result.json"


def run_evaluation():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    results = []
    passed = 0

    print(f"开始评测，共 {len(cases)} 条...\n")

    for case in cases:
        messages = new_session_messages()
        start = time.time()
        try:
            _, answer = chat(messages, case["question"])
            ok = bool(answer and len(answer) > 5)
            if ok:
                passed += 1
            status = "pass" if ok else "fail"
        except Exception as e:
            answer = f"ERROR: {e}"
            status = "error"

        elapsed = time.time() - start
        results.append({
            "id": case["id"],
            "question": case["question"],
            "expect": case["expect"],
            "answer": answer[:500],
            "status": status,
            "latency_sec": round(elapsed, 2),
        })
        print(f"[{case['id']:02d}] {status} ({elapsed:.1f}s) {case['question'][:30]}...")
        if status == "error":
            print(f"       -> {answer[:120]}")

    summary = {
        "total": len(cases),
        "passed": passed,
        "success_rate": round(passed / len(cases) * 100, 1),
        "results": results,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：{passed}/{len(cases)} 通过（{summary['success_rate']}%）")
    print(f"详细结果：{OUT_FILE}")
    return summary


if __name__ == "__main__":
    run_evaluation()
