"""测试 LLM-as-Judge。"""
from agent.graph import build_graph
from eval.judge import judge_report

graph = build_graph()
result = graph.invoke({"pdf_path": "data/papers/test.pdf"})
report = result.get("report")
context = result.get("retrieved_context", "")

if report:
    score = judge_report(report, context)
    print("=== 评测结果 ===")
    print(f"完整性: {score.completeness}")
    print(f"准确性: {score.accuracy}")
    print(f"可操作性: {score.actionability}")
    print(f"风险识别: {score.risk_identification}")
    print(f"综合: {score.overall}")
    print(f"理由: {score.reasoning}")
else:
    print("报告生成失败")
