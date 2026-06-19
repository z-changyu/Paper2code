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

    # TODO(Day4): 在 prompt 里写清楚 ReproReport 的 JSON schema 要求
    system = "你是一个 ML 论文复现专家，只输出 JSON，不要任何额外文字。"
    user = (
        f"复现规划：\n{plan}\n\n"
        f"相关论文片段：\n{context or paper[:6000]}\n\n"
        "请输出符合以下结构的 JSON：title, summary, model_skeleton, "
        "hyperparameters(列表，每项含 name/value/source), risks(字符串列表)。"
    )
    raw = llm.chat(system, user, max_tokens=2048)

    # Day 4: 用 Pydantic 解析校验
    report = None
    try:
        cleaned = llm.extract_json(raw)
        report = ReproReport.model_validate_json(cleaned)
    except Exception as e:
        print(f"[executor] JSON 解析失败，先返回原始文本。错误: {e}")

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
