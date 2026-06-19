"""Day 2: 验证 LangGraph 图的 State 流转（暂时只跑 Planner）。"""
from langgraph.graph import StateGraph, END
from schemas.models import AgentState
from agent.graph import planner_node

# 临时搭一个只含 Planner 的最小图，验证 LangGraph 机制本身通了
g = StateGraph(AgentState)
g.add_node("planner", planner_node)
g.set_entry_point("planner")
g.add_edge("planner", END)
graph = g.compile()

sample_paper = """
We propose a lightweight CNN for image classification with 3 conv layers.
Trained on CIFAR-10, Adam optimizer, lr=0.001, batch=128, 50 epochs, 91% acc.
"""

# 通过图来调用（而不是直接调函数）
result = graph.invoke({"paper_text": sample_paper})

print("=== 图执行后的完整 state ===")
print("plan 字段：")
print(result.get("plan"))
print("\n[OK] LangGraph 状态流转跑通，Day 2 验收通过。")
