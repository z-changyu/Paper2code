"""
LangGraph 编排逻辑 —— Day 2-3 实现，第二周升级。

第一周目标：最简串行图  Planner -> Executor -> END
第二周升级：加入 Critic 节点 + 条件边（不合格打回重做）+ PostgresSaver 持久化

下面是 Day 2-3 的骨架，已写好结构和注释，你填充 TODO 即可。
"""
from langgraph.graph import StateGraph, END
from schemas.models import AgentState, ReproReport
from serving.llm_client import llm
from retrieval.advanced_retriever import AdvancedRetriever

# 全局检索器（避免每次重新加载模型）
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = AdvancedRetriever()
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

    critique = state.get("critique", "")
    retry_count = state.get("retry_count", 0)
    if critique:
        base_user += f"\n\n上一版报告存在以下问题，请针对性修正：{critique}"
    MAX_RETRIES = 3
    raw = ""
    report = None 
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
    return {"report": report, "raw_output": raw}

def critic_node(state: AgentState) -> dict:
    report = state.get("report")
    retry_count = state.get("retry_count", 0)   # 读当前计数
    problems = []

    if report is None:
        problems.append("报告解析失败，未能生成有效的结构化输出")
    else:
        if not report.model_skeleton or len(report.model_skeleton.strip()) < 20:
            problems.append("model_skeleton 内容过短或缺失，需要补充模型骨架代码")
        if len(report.hyperparameters) == 0:
            problems.append("hyperparameters 为空，需要从论文中提取超参配置")
        if len(report.risks) == 0:
            problems.append("risks 为空，需要列出复现风险点")

    if problems:
        critique = "；".join(problems)
        print(f"[critic] 不通过: {critique}")
        # 不通过 → 计数加一并写回
        return {"critique": critique, "retry_count": retry_count + 1}
    else:
        print("[critic] 通过")
        return {"critique": ""}


MAX_CRITIC_RETRIES = 2  # 最多因 Critic 不通过而重试 2 次

def route_after_critic(state: AgentState) -> str:
    """根据评判结果和重试次数，决定下一步。"""
    critique = state.get("critique", "")
    retry_count = state.get("retry_count", 0)

    if not critique:
        # 没有问题，通过
        return "pass"
    if retry_count >= MAX_CRITIC_RETRIES:
        # 达到重试上限，强制结束（熔断）
        print(f"[route] 已达重试上限 {MAX_CRITIC_RETRIES}，强制结束")
        return "pass"
    # 有问题且未到上限，回 Executor 重做
    print(f"[route] 第 {retry_count + 1} 次重试，打回 Executor")
    return "retry"

# ---------- 组图 ----------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retriever", retriever_node)   
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node) 

    g.set_entry_point("retriever")            # 入口改成 retriever
    g.add_edge("retriever", "planner")        # 检索→规划
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")

    # 条件边：critic 之后根据路由函数分流
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "pass": END,           # 通过 → 结束
            "retry": "executor",   # 不通过 → 回 executor 重做
        },
    )
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
