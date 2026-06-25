"""验证裁判的区分力：故意造一份烂报告 + 一份好报告，看裁判能否区分。"""
from schemas.models import ReproReport, Hyperparameter
from eval.judge import judge_report

# 假装这是某篇 CNN 图像分类论文的相关内容
context = """
We propose a lightweight CNN for image classification with 3 conv layers
and 2 FC layers. Trained on CIFAR-10 with Adam optimizer, learning rate 0.001,
batch size 128, 50 epochs, achieving 91% test accuracy.
"""

# ---------- 1. 一份明显很烂的报告 ----------
bad_report = ReproReport(
    title="未知",
    summary="一个模型",
    model_skeleton="用神经网络",          # 极度敷衍、无法指导复现
    hyperparameters=[],                    # 完全没提取超参
    risks=[],                              # 完全没风险分析
)

# ---------- 2. 一份较好的报告 ----------
good_report = ReproReport(
    title="Lightweight CNN for CIFAR-10 Image Classification",
    summary="3层卷积+2层全连接的轻量CNN，在CIFAR-10上做图像分类",
    model_skeleton=(
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.conv1 = nn.Conv2d(3,32,3); self.conv2 = nn.Conv2d(32,64,3)\n"
        "        self.conv3 = nn.Conv2d(64,128,3)\n"
        "        self.fc1 = nn.Linear(128*2*2,256); self.fc2 = nn.Linear(256,10)\n"
        "    # 3 conv + 2 fc，与论文一致"
    ),
    hyperparameters=[
        Hyperparameter(name="learning_rate", value="0.001", source="论文训练设置"),
        Hyperparameter(name="batch_size", value="128", source="论文训练设置"),
        Hyperparameter(name="epochs", value="50", source="论文训练设置"),
        Hyperparameter(name="optimizer", value="Adam", source="论文训练设置"),
    ],
    risks=[
        "论文未说明权重初始化方式，可能影响复现精度",
        "数据增强策略未明确，91%准确率可能依赖特定预处理",
    ],
)

print("=== 烂报告评分 ===")
bad = judge_report(bad_report, context)
if bad:
    print(f"综合分: {bad.overall} | 完整性:{bad.completeness} 准确性:{bad.accuracy} "
          f"可操作性:{bad.actionability} 风险:{bad.risk_identification}")
    print(f"理由: {bad.reasoning}")

print("\n=== 好报告评分 ===")
good = judge_report(good_report, context)
if good:
    print(f"综合分: {good.overall} | 完整性:{good.completeness} 准确性:{good.accuracy} "
          f"可操作性:{good.actionability} 风险:{good.risk_identification}")
    print(f"理由: {good.reasoning}")

print("\n=== 区分力判断 ===")
if bad and good:
    gap = good.overall - bad.overall
    print(f"好报告 - 烂报告 = {gap:+.2f}")
    if gap >= 1.0:
        print("[OK] 裁判有区分力，评测体系可信")
    else:
        print("[警告] 区分度太小，裁判可能失效，需要调整评分 prompt")
