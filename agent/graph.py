"""
LangGraph 编排逻辑 —— Day 2-3 实现，第二周升级。

第一周目标：最简串行图  Planner -> Executor -> END
第二周升级：加入 Critic 节点 + 条件边（不合格打回重做）+ PostgresSaver 持久化

下面是 Day 2-3 的骨架，已写好结构和注释，你填充 TODO 即可。
"""
from langgraph.graph import StateGraph, END
from schemas.models import AgentState, ReproReport
from serving.llm_client import llm
from retrieval.retriever import SimpleRetriever

# 全局检索器（避免每次重新加载模型）
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = SimpleRetriever()
    return _retriever


def retriever_node(state: AgentState) -> dict:
    """加载 PDF、检索关键内容，写入 retrieved_context。"""
    pdf_path = state.get("pdf_path")
    if not pdf_path:
        # 没给 PDF 就跳过检索，用已有的 paper_text
        return {}

    r = get_retriever()
    r.load_pdf(pdf_path)

    # 用几个面向"复现"的查询，把关键信息都检索出来拼一起
    queries = [
        "model architecture and network structure",
        "training hyperparameters learning rate batch size epochs",
        "dataset and experimental setup",
    ]
    chunks = []
    for q in queries:
        chunks.append(r.search(q, top_k=2))
    context = "\n---\n".join(chunks)

    # 同时把检索到的内容也作为 paper_text（供 Planner 用）
    return {"retrieved_context": context, "paper_text": context}

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
    """生成结构化复现报告，带校验和重试。"""
    plan = state.get("plan", "")
    paper = state["paper_text"]
    context = state.get("retrieved_context", "")

    system = (
        "你是一个 ML 论文复现专家，输出结构化复现报告。"
        "严格只输出一个 JSON 对象，不要解释文字，不要 markdown 代码块标记。"
    )
    base_user = f"""根据以下规划和论文内容，生成结构化复现报告。

复现规划：
{plan}

论文内容：
{context or paper[:6000]}

请输出符合以下结构的 JSON（只输出 JSON）：
{{
  "title": "论文标题",
  "summary": "方法一句话概述",
  "model_skeleton": "模型骨架代码（必须有内容）",
  "hyperparameters": [{{"name": "learning_rate", "value": "0.001", "source": "第4节"}}],
  "risks": ["风险点1", "风险点2"]
}}"""

    MAX_RETRIES = 3
    raw = ""
    feedback = ""  # 上一轮的错误反馈

    for attempt in range(1, MAX_RETRIES + 1):
        user = base_user
        if feedback:
            user += f"\n\n注意：你上一次的输出有问题：{feedback}。请修正后重新输出完整 JSON。"

        raw = llm.chat(system, user, max_tokens=2048)
        try:
            cleaned = llm.extract_json(raw)
            report = ReproReport.model_validate_json(cleaned)
            print(f"[executor] 第 {attempt} 次尝试成功")
            return {"report": report, "raw_output": raw}
        except Exception as e:
            feedback = str(e)
            print(f"[executor] 第 {attempt} 次失败: {feedback}")

    # 重试上限耗尽，返回失败信号
    print(f"[executor] {MAX_RETRIES} 次尝试均失败，放弃")
    return {"report": None, "raw_output": raw}

# ---------- 组图 ----------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retriever", retriever_node)   # 新增
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)

    g.set_entry_point("retriever")            # 入口改成 retriever
    g.add_edge("retriever", "planner")        # 检索→规划
    g.add_edge("planner", "executor")
    g.add_edge("executor", END)
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
