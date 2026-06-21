"""Day 3: 端到端测试完整图 Planner -> Executor。"""
from agent.graph import build_graph

graph = build_graph()

sample_paper = """
We propose a lightweight CNN for image classification with 3 conv layers
followed by 2 FC layers. Trained on CIFAR-10 with Adam optimizer,
learning rate 0.001, batch size 128, 50 epochs, achieving 91% test accuracy.
"""

result = graph.invoke({"paper_text": sample_paper})

print("=== PLAN（Planner产出）===")
print(result.get("plan"))
print("\n=== REPORT（Executor产出）===")
report = result.get("report")
if report:
    print(f"标题: {report.title}")
    print(f"概述: {report.summary}")
    print(f"模型骨架: {report.model_skeleton[:200]}...")
    print(f"超参数量: {len(report.hyperparameters)}")
    print(f"风险点数: {len(report.risks)}")
    print("\n[OK] 端到端链路跑通，Day 3 验收通过！")
else:
    print("report 解析失败，原始输出：")
    print(result.get("raw_output", "")[:500])
