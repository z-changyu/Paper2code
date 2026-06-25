"""Pairwise 评测:让裁判直接对比两份报告哪个更好(含位置偏见消除)。"""
from serving.llm_client import llm


def _judge_once(report_a_text: str, report_b_text: str, context: str) -> str:
    """单次对比,返回 'A' / 'B' / 'TIE'。"""
    system = (
        "你是严格的ML论文复现报告评审专家。对比两份报告,判断哪份质量更高。"
        "只输出一个词:A 或 B 或 TIE,不要解释。"
    )
    user = f"""论文相关内容:
{context[:2500]}

报告A:
{report_a_text}

报告B:
{report_b_text}

哪份报告整体质量更高(更完整、超参更准、骨架更可用、风险识别更到位)?
只回答 A、B 或 TIE。"""
    out = llm.chat(system, user, max_tokens=10).strip().upper()
    if "A" in out and "B" not in out:
        return "A"
    if "B" in out and "A" not in out:
        return "B"
    return "TIE"


def _report_to_text(r) -> str:
    if r is None:
        return "(无有效报告)"
    return (f"概述:{r.summary}\n模型骨架:{r.model_skeleton[:400]}\n"
            f"超参:{[(h.name,h.value) for h in r.hyperparameters]}\n风险:{r.risks}")


def pairwise_compare(agent_report, base_report, context: str) -> str:
    """对比 agent vs baseline,两个顺序各评一次消除位置偏见。
    返回 'AGENT' / 'BASELINE' / 'TIE'。"""
    a_text = _report_to_text(agent_report)
    b_text = _report_to_text(base_report)

    # 顺序1: agent=A, baseline=B
    r1 = _judge_once(a_text, b_text, context)
    # 顺序2: baseline=A, agent=B（交换位置）
    r2 = _judge_once(b_text, a_text, context)

    # 把两次结果翻译成"谁赢"
    win1 = "AGENT" if r1 == "A" else ("BASELINE" if r1 == "B" else "TIE")
    win2 = "BASELINE" if r2 == "A" else ("AGENT" if r2 == "B" else "TIE")

    # 两次一致才算数,否则平局
    if win1 == win2:
        return win1
    return "TIE"