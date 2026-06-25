
import asyncio
from agent.graph import build_graph, _mcp_retrieve
from eval.baseline import naive_rag_report
from eval.pairwise_judge import pairwise_compare
from eval.dataset import EVAL_SET
def run_one(pdf_path):
    context = asyncio.run(_mcp_retrieve(pdf_path, ["model", "hyperparameters", "dataset"]))
    agent_report = build_graph().invoke({"pdf_path": pdf_path}).get("report")
    base_report = naive_rag_report(pdf_path)
    winner = pairwise_compare(agent_report, base_report, context)
    return winner

def main():
    from eval.dataset import EVAL_SET
    wins = {"AGENT": 0, "BASELINE": 0, "TIE": 0}
    for item in EVAL_SET:
        print(f"\n评测: {item['domain']} - {item['pdf']}")
        try:
            w = run_one(item["pdf"])
            wins[w] += 1
            print(f"  胜者: {w}")
        except Exception as e:
            print(f"  出错: {e}")
    print("\n========== Pairwise 汇总 ==========")
    print(f"Agent 胜: {wins['AGENT']}")
    print(f"Baseline 胜: {wins['BASELINE']}")
    print(f"平局: {wins['TIE']}")

if __name__ == "__main__":
    main()