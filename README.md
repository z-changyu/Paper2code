# Paper2Code · 论文复现 Agent 系统

> 一个将 ML 论文自动转化为结构化复现报告（模型骨架 + 超参配置 + 风险清单）的 Agent 系统。
> 基于 LangGraph 编排、vLLM 本地推理、RAG 检索增强，将论文复现的前置调研时间从数天压缩至分钟级。

<!-- 建议在此处放一张 /generate_report 接口返回结果或 /docs 页面的截图 -->

---

## 项目简介

复现一篇 ML 论文往往要花数天做前置调研：超参分散在正文各处、官方代码常缺失、模型结构需要反复推敲。Paper2Code 通过一个多节点 Agent，自动阅读论文并产出结构化的复现报告，包含模型骨架代码、超参配置（含出处）和复现风险清单。

系统采用多服务解耦架构：Agent 编排、模型推理、检索能力各为独立服务，通过标准接口（HTTP / MCP 协议）通信，各自独立管理依赖与扩展。

---

## 系统架构

```
                    ┌─────────────────────────────┐
   用户/客户端 ──HTTP──>│  Agent API (FastAPI, 8080)   │  ← 统一入口（已容器化）
                    │  LangGraph 状态图编排：       │
                    │  retriever → planner →       │
                    │  executor → critic（带循环）  │
                    └──────┬──────────────┬────────┘
                           │ MCP(SSE)     │ HTTP(OpenAI兼容)
                           ▼              ▼
              ┌─────────────────┐  ┌──────────────────┐
              │ MCP Server      │  │ vLLM 推理服务     │
              │ (SSE, 检索环境) │  │ (主环境)          │
              │ Docling 解析    │  │ Qwen2.5 + NF4量化 │
              │ BGE-M3 向量化   │  │ 单卡 RTX 3090     │
              │ Qdrant 向量库   │  │ PagedAttention    │
              └─────────────────┘  └──────────────────┘
```

**设计要点**：推理服务（依赖旧版 transformers）和检索服务（依赖新版 transformers + Docling）存在依赖冲突，因此物理隔离为两个独立环境/进程，通过网络接口解耦通信——这既解决了依赖冲突，也实现了模型推理与数据处理的职责分离。

---

## 核心特性

- **多节点 Agent 编排**：基于 LangGraph StateGraph 构建 Planner-Executor-Critic 流程，Critic 节点对生成报告做质量校验，不合格则带反馈打回重做，并设重试上限防止死循环。
- **本地大模型量化部署**：单卡 RTX 3090 上通过 NF4 量化部署 Qwen2.5 系列大模型，结合 PagedAttention、prefix caching 优化显存（注：FP8 KV cache 因 Ampere 架构不支持而未使用，改用 FP16 KV cache）。
- **结构化输出保障**：Pydantic schema 强约束 + JSON 清洗 + 带反馈重试，应对 LLM 输出的不确定性，保证下游可稳定解析。
- **RAG 检索增强**：Docling 解析 PDF → BGE-M3 向量化 → Qdrant 向量库存储与检索。
- **服务解耦**：检索能力通过独立服务暴露，提供 FastAPI（REST）和 MCP server（SSE transport）两种接口，编排层远程调用。
- **评测体系**：LLM-as-Judge 多维度评分 + pairwise 对比评测（含位置偏见消除），与 naive RAG baseline 对比验证架构效果。

---

## 技术栈

| 层次 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| 模型推理 | vLLM、Qwen2.5、NF4 量化（bitsandbytes） |
| 检索 (RAG) | Docling、BGE-M3、Qdrant |
| 服务/接口 | FastAPI、MCP (SSE transport) |
| 数据校验 | Pydantic |
| 部署 | Docker |
| 评测 | LLM-as-Judge、pairwise comparison |

---

## 目录结构

```
paper2code/
├── agent/          # LangGraph 编排逻辑（节点、状态图）
├── serving/        # 模型调用封装、FastAPI 服务、MCP server
├── retrieval/      # PDF 解析与检索
├── schemas/        # Pydantic 数据模型
├── eval/           # 评测体系（judge、baseline、批量评测）
├── tests/          # 测试脚本
├── data/papers/    # 测试论文
├── Dockerfile      # Agent API 服务容器化
└── requirements*.txt
```

---

## 快速开始

系统由三个服务组成，需分别启动（推理与检索在独立环境）。

### 1. 启动 vLLM 推理服务（主环境）

```bash
conda activate paper2code
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/start_vllm_qwen14b.sh   # 单卡 NF4 量化部署 Qwen2.5
```

### 2. 启动检索 MCP server（检索环境）

```bash
conda activate paper2code-rag
export HF_ENDPOINT=https://hf-mirror.com
python serving/mcp_server.py          # SSE transport，暴露检索工具
```

### 3. 启动 Agent API（主环境）

```bash
conda activate paper2code
uvicorn serving.agent_api:app --host 0.0.0.0 --port 8080
# 或使用 Docker：
# docker build -t paper2code-agent-api .
# docker run -d --name agent-api --network host paper2code-agent-api
```

### 4. 调用

```bash
curl -X POST http://localhost:8080/generate_report \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "data/papers/test.pdf"}'
```

也可访问 `http://localhost:8080/docs` 使用交互式 API 文档。

---

## 评测

采用 LLM-as-Judge 对复现报告做多维度（完整性、准确性、可操作性、风险识别）评分，并与 naive RAG baseline（仅检索 + 一次生成，无规划/评判/重试）对比。

评测过程中发现：绝对打分方式区分度不足（分数集中），改用 **pairwise 对比评测 + 位置偏见消除**（两个顺序各评一次，一致才计）后得到更可信的结论。

在 6 篇多领域（CV / NLP / RL 等）评测集上的 pairwise 结果：

| 对比 | 结果 |
|------|------|
| 完整 Agent vs naive RAG | **Agent 2 胜 / 0 负 / 4 平** |

结论：Agent 架构相比 naive RAG **未现劣势，并在部分样本上更优**。在方法清晰的经典论文上两者差异不大；Agent 的规划与评判机制在更复杂、信息分散的任务上预期收益更明显。

> 运行评测：`python -m eval.run_eval`

---

## 已知限制与后续优化

- **PDF 解析**：受环境约束，部分流程使用 pypdf 解析（Docling 已在独立检索环境跑通）；pypdf 解析未区分正文与参考文献，检索召回偶有文献条目噪声，后续可用 Docling 的结构化解析过滤。
- **评测集规模**：当前评测集偏小且以经典论文为主，后续可扩充复杂/长尾论文，更充分检验 Agent 架构收益。
- **裁判模型**：受本地资源限制，评测使用与被评者相同的本地模型，存在自评偏差；生产环境宜使用更强的独立裁判模型。
- **容器化范围**：已容器化纯 Python 的 Agent API 服务；vLLM（GPU）服务的容器化涉及 GPU 透传，未包含在当前镜像中。

---

## 工程笔记

本项目在受限硬件（单卡 24GB）与特殊环境下从零搭建，过程中的依赖冲突排查、硬件能力边界判断、评测方法迭代等记录见 [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md)。
