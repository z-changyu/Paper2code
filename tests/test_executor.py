"""Day 3: 单独测试 Executor 节点。"""
from agent.graph import executor_node
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sample_paper = """
We propose a lightweight CNN for image classification with 3 conv layers
followed by 2 FC layers. Trained on CIFAR-10 with Adam optimizer,
learning rate 0.001, batch size 128, 50 epochs, achieving 91% test accuracy.
"""
# 手动构造一个 plan（模拟 Planner 的输出）
fake_plan = "1. 搭建3层卷积+2层全连接的CNN  2. 加载CIFAR-10  3. 用Adam(lr=0.001)训练50轮  4. 评估测试准确率"

result = executor_node({"paper_text": sample_paper, "plan": fake_plan})

print("=== report 对象 ===")
print(result["report"])
print("\n=== 如果解析失败，看原始输出 ===")
if result["report"] is None:
    print(result["raw_output"])
