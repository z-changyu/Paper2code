# Paper2Code · 论文复现 Agent 系统

自动将 ML 论文转化为结构化复现报告（模型骨架代码 + 超参配置 + 风险清单）的 Agent 系统。
基于 LangGraph 编排、vLLM 本地推理、RAG 检索增强，将论文复现的前置调研时间从数天压缩至分钟级。

> 开发中。当前进度：第一周（端到端最小闭环）。

## 架构概览

```
论文 PDF
   │
   ▼
[检索层] Docling/pypdf 解析 → 切块 → 向量检索（BGE/Qdrant）
   │
   ▼
[Agent 编排层] LangGraph StateGraph
   Planner（规划复现步骤）
      → Executor（生成结构化报告）
      → Critic（质量校验，不合格打回）   ← 第二周
   │
   ▼
[模型服务层] vLLM 部署 LLM（第一周 Qwen2.5-3B → 第二周 Gemma 4 31B + 量化优化）
   │
   ▼
结构化复现报告（Pydantic schema 约束的 JSON）
```

## 目录结构

```
paper2code/
├── agent/          # LangGraph 编排逻辑（graph.py）
├── serving/        # vLLM 模型服务封装（llm_client.py）
├── retrieval/      # PDF 解析与检索（retriever.py）
├── schemas/        # Pydantic 数据模型（models.py）
├── tools/          # 工具层 / MCP server（第二周）
├── eval/           # 评测体系（第三周）
├── scripts/        # 环境检查、启动、验证脚本
├── tests/          # 测试
└── data/papers/    # 测试论文 PDF 放这里
```

## 快速开始（Day 1）

```bash
# 1. 创建环境
conda create -n paper2code python=3.11 -y
conda activate paper2code

# 2. 装依赖（先装 vllm，它会带上匹配的 torch）
pip install vllm
pip install -r requirements.txt

# 3. 环境自检
python scripts/check_env.py

# 4. 启动 vLLM（占用当前终端，保持开启）
bash scripts/start_vllm.sh

# 5. 新开一个终端，验证连通
python scripts/test_vllm.py
```

看到模型正常回复，即 Day 1 验收通过。

## 开发路线

- **第一周**：端到端最小闭环（小模型 + 最简检索 + 串行 Agent）✅ 进行中
- **第二周**：升级正经技术栈（Gemma 4 31B + 量化、Critic 节点、Docling+BGE-M3+Qdrant、MCP server）
- **第三周**：评测体系（LLM-as-Judge + baseline 对比）、FastAPI + Docker 工程化、文档
