#!/usr/bin/env bash
# Day 1 启动 vLLM 服务（第一周用小模型，先把链路跑通）。
#
# 用法：  bash scripts/start_vllm.sh
# 这个命令会一直占用当前终端（服务在前台运行），
# 所以请新开一个终端运行验证脚本 test_vllm.py。
#
# 第一周特意用 Qwen2.5-3B 这种小模型：
#   - 3090 (24G) 跑 3B 毫无压力，启动快，方便反复调试链路
#   - 31B + 量化那些硬骨头留到第二周 Day 6-7
#
# 关键参数解释：
#   --gpu-memory-utilization 0.85  : vLLM 预占的显存比例，留点余量给系统，避免 OOM
#   --max-model-len 8192           : 单次请求最大上下文长度，小模型设小一点省显存
#   --served-model-name            : 对外暴露的模型名，要和客户端代码里的 MODEL 一致

set -e
export HF_ENDPOINT=https://hf-mirror.com
MODEL="Qwen/Qwen2.5-3B-Instruct"

echo "启动 vLLM 服务，加载模型: $MODEL"
echo "首次运行会自动从 HuggingFace 下载模型，请耐心等待..."
echo "服务地址: http://localhost:8000/v1"
echo "（保持本终端开启，另开终端运行 python scripts/test_vllm.py 验证）"
echo

vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port 8001
