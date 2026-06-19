"""
Day 1 vLLM 连通性验证。
前置：先在另一个终端启动 vLLM 服务（见 scripts/start_vllm.sh）。

运行: python scripts/test_vllm.py

vLLM 启动后会暴露一个 OpenAI 兼容的接口（默认 http://localhost:8000/v1），
所以我们直接用 openai 这个库去调它 —— 这也是为什么后面接 LangChain 很方便，
生态都兼容 OpenAI 格式。
"""
from openai import OpenAI

# vLLM 的 OpenAI 兼容服务地址。api_key 随便填，本地服务不校验。
client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

# 模型名要和启动 vLLM 时 --served-model-name 一致（默认是模型路径）
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("向本地 vLLM 发送测试请求...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "用一句话介绍什么是 KV cache。"},
        ],
        max_tokens=128,
        temperature=0.3,
    )
    print("\n模型回复：")
    print(resp.choices[0].message.content)
    print("\n[OK] vLLM 服务连通，Day 1 验收通过。")


if __name__ == "__main__":
    main()
