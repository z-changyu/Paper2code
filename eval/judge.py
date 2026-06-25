"""LLM-as-Judge 评测器:用 LLM 评估复现报告质量。"""
from serving.llm_client import llm
from schemas.models import JudgeScore, ReproReport


def judge_report(report: ReproReport, paper_context: str) -> JudgeScore:
    """让 LLM 按多维度评估一份复现报告。"""
    system = (
        "你是一个严格的 ML 论文复现质量评审专家。"
        "你将根据论文内容,评估一份复现报告的质量,严格只输出 JSON。"
    )
    user = f"""请评估以下复现报告的质量。

【论文相关内容】
{paper_context[:3000]}

【待评估的复现报告】
标题: {report.title}
概述: {report.summary}
模型骨架: {report.model_skeleton}
超参数: {[(h.name, h.value) for h in report.hyperparameters]}
风险点: {report.risks}

请从四个维度打分(每项1-5分),并给出综合分和理由。评分标准:
- completeness(完整性): 关键信息是否齐全
- accuracy(准确性): 提取的超参/方法是否与论文一致
- actionability(可操作性): 模型骨架能否指导实际复现
- risk_identification(风险识别): 是否点出了关键复现风险

只输出如下 JSON:
{{"completeness":4,"accuracy":3,"actionability":4,"risk_identification":3,"overall":3.5,"reasoning":"理由..."}}"""

    raw = llm.chat(system, user, max_tokens=1024)
    try:
        cleaned = llm.extract_json(raw)
        return JudgeScore.model_validate_json(cleaned)
    except Exception as e:
        print(f"[judge] 解析失败: {e}, 原始: {raw[:300]}")
        return None