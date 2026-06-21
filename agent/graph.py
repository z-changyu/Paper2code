"""
LangGraph 编排逻辑 —— Day 2-3 实现，第二周升级。

第一周目标：最简串行图  Planner -> Executor -> END
第二周升级：加入 Critic 节点 + 条件边（不合格打回重做）+ PostgresSaver 持久化

下面是 Day 2-3 的骨架，已写好结构和注释，你填充 TODO 即可。
"""
from langgraph.graph import StateGraph, END
from schemas.models import AgentState, ReproReport
from serving.llm_client import llm


# ---------- 节点 1: Planner ----------
def planner_node(state: AgentState) -> dict:
    """读取论文文本，规划复现步骤，写回 state['plan']。"""
    paper = state["paper_text"]
    # TODO(Day2): 写 prompt，让模型输出"复现这篇论文需要哪几步"
    system = "你是一个 ML 论文复现规划专家。"
    user = f"阅读以下论文内容，列出复现它需要的关键步骤：\n\n{paper[:6000]}"
    plan = llm.chat(system, user, max_tokens=1024)
    return {"plan": plan}


# ---------- 节点 2: Executor ----------
def executor_node(state: AgentState) -> dict:
    """根据规划和论文内容，生成结构化复现报告。"""
    plan = state.get("plan", "")
    paper = state["paper_text"]
    context = state.get("retrieved_context", "")  # Day 5 检索结果接入

    system = (
        "你是一个 ML 论文复现专家。你的任务是输出一份结构化复现报告。"
        "严格只输出一个 JSON 对象，不要输出任何解释文字，不要使用 markdown 代码块标记。"
    )
    user = f"""根据以下复现规划和论文内容，生成结构化复现报告。

复现规划：
{plan}

论文内容：
{context or paper[:6000]}

请输出符合以下结构的 JSON（只输出 JSON 本身）：
{{
  "title": "论文标题",
  "summary": "方法的一句话概述",
  "model_skeleton": "模型骨架的伪代码或 PyTorch 代码片段",
  "hyperparameters": [
    {{"name": "learning_rate", "value": "0.001", "source": "论文第4节"}}
  ],
  "risks": ["复现风险点1", "复现风险点2"]
}}"""

    raw = llm.chat(system, user, max_tokens=2048)

    report = None
    try:
        cleaned = llm.extract_json(raw)
        report = ReproReport.model_validate_json(cleaned)
    except Exception as e:
        print(f"[executor] JSON 解析失败: {e}")
        print(f"[executor] 原始输出前500字: {raw[:500]}")

    return {"report": report, "raw_output": raw}

# ---------- 组图 ----------
def build_graph():
    """构建并编译 LangGraph 状态图。"""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", END)

    # 第二周：在这里加 critic 节点和条件边
    return g.compile()


if __name__ == "__main__":
    # Day 3 验收：喂一段论文文本，看能否走完全流程
    graph = build_graph()
    sample = "（这里粘贴一段论文摘要+方法部分作为测试输入）"
    result = graph.invoke({"paper_text": sample})
    print("=== PLAN ===")
    print(result.get("plan"))
    print("\n=== REPORT ===")
    print(result.get("report") or result.get("raw_output"))
