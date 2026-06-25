"""naive RAG baseline:检索 + 一次性生成,不用 Agent 的规划/评判/迭代。"""
import asyncio
from agent.graph import _mcp_retrieve  # 复用 MCP 检索
from serving.llm_client import llm
from schemas.models import ReproReport


def naive_rag_report(pdf_path: str) -> ReproReport:
    """最简方法:检索一次,直接让模型一次性生成报告(无规划、无评判、无重试)。"""
    queries = ["model architecture", "training hyperparameters", "dataset setup"]
    context = asyncio.run(_mcp_retrieve(pdf_path, queries))

    system = "你是论文复现专家,只输出 JSON。"
    user = f"""根据以下论文内容,生成结构化复现报告。
论文内容:
{context[:4000]}

请只输出符合以下结构的 JSON(不要任何额外文字):
{{
  "title": "论文标题",
  "summary": "方法一句话概述",
  "model_skeleton": "模型骨架代码",
  "hyperparameters": [{{"name":"learning_rate","value":"0.001","source":"第4节"}}],
  "risks": ["风险点1","风险点2"]
}}"""
    raw = llm.chat(system, user, max_tokens=2048)
    try:
        return ReproReport.model_validate_json(llm.extract_json(raw))
    except Exception as e:
        print(f"[baseline] JSON 解析失败: {e}")
        print(f"[baseline] 原始输出前500字: {raw[:500]}")
        return None