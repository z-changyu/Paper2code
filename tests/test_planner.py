"""Day 2: 单独测试 Planner 节点（不经过图）。"""
from agent.graph import planner_node
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 手动构造一段论文文本作为输入（先用摘要级别的短文本）
sample_paper = """
We propose a method for image classification using a lightweight CNN.
The model uses 3 convolutional layers followed by 2 fully connected layers.
We train on CIFAR-10 with Adam optimizer, learning rate 0.001, batch size 128,
for 50 epochs. We achieve 91% accuracy on the test set.
"""

# 直接调用节点函数，传入一个最小 state
result = planner_node({"paper_text": sample_paper})

print("=== Planner 返回的更新 ===")
print(result)
print("\n=== plan 内容 ===")
print(result["plan"])
