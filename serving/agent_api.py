"""Agent API:把论文复现 Agent 包装成统一的 HTTP 服务。
对外暴露 /generate_report 接口,内部跑 LangGraph(检索调MCP、生成调vLLM)。
这是整个系统的统一入口。
"""
from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import build_graph

app = FastAPI(title="Paper2Code Agent API", description="论文复现报告生成服务")

# 启动时编译一次图(避免每个请求重新编译)
graph = build_graph()


class GenerateRequest(BaseModel):
    pdf_path: str


@app.post("/generate_report")
def generate_report(req: GenerateRequest):
    """输入PDF路径,返回结构化复现报告。"""
    result = graph.invoke({"pdf_path": req.pdf_path})
    report = result.get("report")
    if report is None:
        return {"status": "failed", "raw_output": result.get("raw_output", "")[:500]}
    return {
        "status": "ok",
        "report": report.model_dump(),  # Pydantic 模型转 dict,FastAPI 自动序列化为 JSON
    }


@app.get("/health")
def health():
    return {"status": "alive"}  