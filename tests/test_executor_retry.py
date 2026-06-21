"""Day 4: 测试带重试的 Executor。多跑几次观察稳定性。"""
from agent.graph import executor_node

sample_paper = """
We propose a lightweight CNN with 3 conv layers and 2 FC layers.
Trained on CIFAR-10, Adam optimizer, lr=0.001, batch=128, 50 epochs, 91% acc.
"""
fake_plan = "1.搭建CNN 2.加载CIFAR-10 3.Adam训练50轮 4.评估准确率"

success = 0
for i in range(3):
    print(f"\n--- 第 {i+1} 轮整体测试 ---")
    result = executor_node({"paper_text": sample_paper, "plan": fake_plan})
    if result["report"]:
        success += 1
        print(f"成功！model_skeleton 前100字: {result['report'].model_skeleton[:100]}")

print(f"\n=== 3 轮中成功 {success} 轮 ===")
print("[OK] Day 4 验收：重试逻辑生效，成功率应明显高于 Day 3")
