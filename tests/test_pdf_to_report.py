"""Day 5: 第一周终极测试 —— PDF 文件 → 结构化复现报告。"""
from agent.graph import build_graph

graph = build_graph()

# 只传 PDF 路径，其余全自动
result = graph.invoke({"pdf_path": "data/papers/test.pdf"})

print("=== 检索到的内容（前300字）===")
print((result.get("retrieved_context") or "")[:300])

print("\n=== PLAN ===")
print((result.get("plan") or "")[:400])

print("\n=== REPORT ===")
report = result.get("report")
if report:
    print(f"标题: {report.title}")
    print(f"概述: {report.summary}")
    print(f"模型骨架(前200字): {report.model_skeleton[:200]}")
    print(f"超参数: {[(h.name, h.value) for h in report.hyperparameters]}")
    print(f"风险点: {report.risks}")
    print("\n🎉 [OK] PDF→报告 端到端跑通！第一周保命链路闭合！")
else:
    print("report 解析失败，原始输出：")
    print((result.get("raw_output") or "")[:500])
