"""
Day 1 环境自检脚本。
运行: python scripts/check_env.py
逐项检查开发环境是否就绪，每一项都会打印 [OK] / [WARN] / [FAIL]。
看到任何 [FAIL] 先解决了再往下做。
"""
import sys
import shutil
import subprocess


def check(label, ok, detail=""):
    tag = "[OK]  " if ok else "[FAIL]"
    print(f"{tag} {label}" + (f"  ->  {detail}" if detail else ""))
    return ok


def warn(label, detail=""):
    print(f"[WARN] {label}" + (f"  ->  {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("Paper2Code 环境自检")
    print("=" * 60)

    # 1. Python 版本 (vLLM 需要 3.9+，推荐 3.10/3.11)
    v = sys.version_info
    check(
        f"Python 版本 {v.major}.{v.minor}.{v.micro}",
        v >= (3, 10),
        "建议 3.10 或 3.11" if v < (3, 10) else "",
    )

    # 2. nvidia-smi 是否可用 (能看到 GPU)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                text=True,
            ).strip()
            check("检测到 GPU", True, out)
        except Exception as e:
            check("nvidia-smi 执行失败", False, str(e))
    else:
        check("找到 nvidia-smi", False, "未检测到 NVIDIA 驱动，确认在带卡的机器上运行")

    # 3. PyTorch + CUDA 是否可用
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        check(f"PyTorch {torch.__version__}", True)
        check(
            "PyTorch CUDA 可用",
            cuda_ok,
            f"设备: {torch.cuda.get_device_name(0)}" if cuda_ok else "torch 看不到 GPU，多半是 CUDA 版本不匹配",
        )
    except ImportError:
        warn("PyTorch 未安装", "安装 vLLM 时会自动带上，先装 vLLM")

    # 4. vLLM 是否安装
    try:
        import vllm
        check(f"vLLM {vllm.__version__}", True)
    except ImportError:
        check("vLLM 未安装", False, "运行: pip install vllm")

    # 5. 其他第一周依赖
    for mod, hint in [
        ("langgraph", "pip install langgraph"),
        ("langchain_core", "pip install langchain-core"),
        ("pydantic", "pip install pydantic"),
        ("openai", "pip install openai  # 用于调用 vLLM 的 OpenAI 兼容接口"),
        ("sentence_transformers", "pip install sentence-transformers"),
        ("pypdf", "pip install pypdf"),
        ("numpy", "pip install numpy"),
    ]:
        try:
            __import__(mod)
            check(f"{mod}", True)
        except ImportError:
            check(f"{mod} 未安装", False, hint)

    print("=" * 60)
    print("自检完成。所有 [FAIL] 解决后即可进入 Day 2。")
    print("=" * 60)


if __name__ == "__main__":
    main()
