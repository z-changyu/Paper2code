"""
LLM 调用的统一封装。

所有节点都通过这里调用模型，好处是：
- 换模型 / 换部署方式只改这一处（第二周从 3B 换 31B，节点代码不用动）
- 统一处理重试、超时、JSON 清洗等通用逻辑

第一周通过 OpenAI 兼容接口调本地 vLLM。
"""
import re
from openai import OpenAI

# 与 start_vllm.sh 里的 --served-model-name 保持一致
DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
BASE_URL = "http://localhost:8001/v1"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = BASE_URL):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def chat(self, system: str, user: str, max_tokens: int = 2048,
             temperature: float = 0.3) -> str:
        """最基础的对话调用，返回纯文本。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    @staticmethod
    def extract_json(text: str) -> str:
        """从模型输出里剥离 JSON。

        LLM 常把 JSON 包在 ```json ... ``` 代码块里，或前后带解释文字。
        这里做最朴素的清洗：优先取代码块内容，否则截取第一个 { 到最后一个 }。
        （第二周上 xgrammar guided decoding 后可从根上保证格式，这里就成了兜底。）
        """
        # 先试代码块
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        # 再试裸 JSON
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text


# 一个模块级单例，方便各节点直接 import 使用
llm = LLMClient()
