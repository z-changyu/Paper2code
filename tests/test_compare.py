"""对比:完整 Agent vs naive RAG baseline。"""
import asyncio
from agent.graph import build_graph, _mcp_retrieve
from eval.judge import judge_report
from eval.baseline import naive_rag_report

PDF = "data/papers/test.pdf"

# 用于评测的检索上下文
context = asyncio.run(_mcp_retrieve(PDF, ["model", "hyperparameters", "dataset"]))

# 1) 完整 Agent
print("=== 完整 Agent ===")
agent_report = build_graph().invoke({"pdf_path": PDF}).get("report")
agent_score = judge_report(agent_report, context) if agent_report else None

# 2) naive baseline
print("=== naive RAG baseline ===")
base_report = naive_rag_report(PDF)
base_score = judge_report(base_report, context) if base_report else None

# 对比
print("\n=== 对比结果 ===")
if agent_score and base_score:
    print(f"完整 Agent  综合分: {agent_score.overall}")
    print(f"naive RAG   综合分: {base_score.overall}")
    print(f"提升: {agent_score.overall - base_score.overall:+.2f}")
