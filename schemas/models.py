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
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Hyperparameter(BaseModel):
    # 容忍模型多吐字段，不报错
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="超参名称")
    value: str = Field(description="取值，统一用字符串")
    source: str = Field(default="", description="出处，可选")

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v):
        # 模型有时给数字，自动转成字符串，避免类型校验失败
        return str(v) if v is not None else ""


class ReproReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="", description="论文标题")
    summary: str = Field(default="", description="方法概述")
    model_skeleton: str = Field(description="模型骨架代码（关键字段，必填）")
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    @field_validator("model_skeleton")
    @classmethod
    def skeleton_not_empty(cls, v):
        # 关键字段做质量校验：模型骨架不能为空
        if not v or not v.strip():
            raise ValueError("model_skeleton 不能为空")
        return v



# ---------- 2. LangGraph 的共享状态 ----------

class AgentState(TypedDict, total=False):
    """贯穿整个 Agent 流程的状态。

    每个节点读取它、更新它、传给下一个节点。
    total=False 表示字段都是可选的（流程推进中逐步填充）。

    第一周字段保持精简，第二周加 Critic 后会再加 'critique'、'retry_count' 等。
    """
    pdf_path: str  
    paper_text: str             # 输入：论文全文（Day 2-3 先手动喂文本，Day 5 接 PDF）
    retrieved_context: str      # 检索到的相关片段（Day 5 填充）
    plan: str                   # Planner 节点产出：复现步骤规划
    report: Optional[ReproReport]  # Executor 节点产出：最终结构化报告
    raw_output: str             # Executor 的原始文本输出（调试用，解析失败时看这个）
