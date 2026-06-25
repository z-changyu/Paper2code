"""批量评测:对评测集每篇论文,对比 Agent vs baseline,汇总统计。"""
import asyncio
from agent.graph import build_graph, _mcp_retrieve
from eval.judge import judge_report
from eval.baseline import naive_rag_report
from eval.dataset import EVAL_SET


def run_one(pdf_path):
    """对单篇论文,跑 Agent 和 baseline,各自评分。"""
    # 评测用的检索上下文
    context = asyncio.run(_mcp_retrieve(pdf_path, ["model", "hyperparameters", "dataset"]))

    # 完整 Agent
    agent_report = build_graph().invoke({"pdf_path": pdf_path}).get("report")
    agent_score = judge_report(agent_report, context) if agent_report else None

    # naive baseline
    base_report = naive_rag_report(pdf_path)
    base_score = judge_report(base_report, context) if base_report else None

    return agent_score, base_score


def main():
    results = []
    for item in EVAL_SET:
        pdf, domain = item["pdf"], item["domain"]
        print(f"\n===== 评测: {domain} - {pdf} =====")
        try:
            agent_s, base_s = run_one(pdf)
            results.append({
                "domain": domain,
                "agent_overall": agent_s.overall if agent_s else None,
                "base_overall": base_s.overall if base_s else None,
                "agent_ok": agent_s is not None,
                "base_ok": base_s is not None,
            })
        except Exception as e:
            print(f"  该篇评测出错: {e}")
            results.append({"domain": domain, "agent_overall": None,
                            "base_overall": None, "agent_ok": False, "base_ok": False})

    # ---- 汇总统计 ----
    print("\n\n========== 汇总 ==========")
    n = len(results)
    agent_success = sum(1 for r in results if r["agent_ok"])
    base_success = sum(1 for r in results if r["base_ok"])
    print(f"总篇数: {n}")
    print(f"Agent 成功率: {agent_success}/{n}")
    print(f"Baseline 成功率: {base_success}/{n}")

    # 两者都成功的篇目,比较平均分
    both = [r for r in results if r["agent_ok"] and r["base_ok"]]
    if both:
        agent_avg = sum(r["agent_overall"] for r in both) / len(both)
        base_avg = sum(r["base_overall"] for r in both) / len(both)
        print(f"\n(在两者都成功的 {len(both)} 篇上)")
        print(f"Agent 平均综合分: {agent_avg:.2f}")
        print(f"Baseline 平均综合分: {base_avg:.2f}")
        print(f"平均提升: {agent_avg - base_avg:+.2f}")

    # 逐篇明细
    print("\n逐篇明细:")
    for r in results:
        print(f"  [{r['domain']}] Agent={r['agent_overall']} Baseline={r['base_overall']}")


if __name__ == "__main__":
    main()