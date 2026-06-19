"""
项目的核心数据结构定义。

这里放两类东西：
1. ReproReport —— 系统最终要产出的"结构化复现报告"的 schema（Day 4 用）
2. AgentState  —— 贯穿 LangGraph 整个流程的共享状态（Day 2 用）

为什么用 Pydantic？
LLM 默认输出自由文本，下游无法稳定解析。用 Pydantic 定义 schema 后，
可以用 ReproReport.model_validate_json() 校验模型输出，校验通过才算合格，
不通过就能捕获错误、触发重试（重试逻辑第二周加）。这就是简历里
"Pydantic schema 强约束输出" 的实质。
"""
from typing import TypedDict, Optional
from pydantic import BaseModel, Field


# ---------- 1. 最终产出物的 schema ----------

class Hyperparameter(BaseModel):
    """单条超参配置。"""
    name: str = Field(description="超参名称，如 learning_rate")
    value: str = Field(description="取值，如 1e-4。统一用字符串避免类型混乱")
    source: str = Field(default="", description="来自论文哪部分，如 '4.2节' 或 'Table 3'")


class ReproReport(BaseModel):
    """论文复现报告 —— 系统的最终输出。

    Day 4 让 Executor 节点按这个 schema 输出 JSON。
    """
    title: str = Field(description="论文标题")
    summary: str = Field(description="方法的一句话概述")
    model_skeleton: str = Field(description="模型骨架代码（伪代码或 PyTorch 骨架）")
    hyperparameters: list[Hyperparameter] = Field(
        default_factory=list, description="从论文中提取的超参配置"
    )
    risks: list[str] = Field(
        default_factory=list, description="复现风险清单，如 '论文未说明数据预处理细节'"
    )


# ---------- 2. LangGraph 的共享状态 ----------

class AgentState(TypedDict, total=False):
    """贯穿整个 Agent 流程的状态。

    每个节点读取它、更新它、传给下一个节点。
    total=False 表示字段都是可选的（流程推进中逐步填充）。

    第一周字段保持精简，第二周加 Critic 后会再加 'critique'、'retry_count' 等。
    """
    paper_text: str             # 输入：论文全文（Day 2-3 先手动喂文本，Day 5 接 PDF）
    retrieved_context: str      # 检索到的相关片段（Day 5 填充）
    plan: str                   # Planner 节点产出：复现步骤规划
    report: Optional[ReproReport]  # Executor 节点产出：最终结构化报告
    raw_output: str             # Executor 的原始文本输出（调试用，解析失败时看这个）
