#!/usr/bin/env bash
# Day 6: 单卡 3090 部署 Qwen2.5-14B AWQ 量化版
set -e
export CUDA_VISIBLE_DEVICES=1

MODEL_PATH="./models/Qwen2.5-14B-Instruct-AWQ"
SERVE_NAME="Qwen2.5-14B-Instruct"

echo "单卡部署: $MODEL_PATH（AWQ量化）"
echo

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVE_NAME" \
  --quantization awq_marlin \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --port 8001
