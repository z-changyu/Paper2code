# Agent API 服务的容器镜像（纯 Python，不含 GPU/大模型，那些作为外部服务通过网络调用）
FROM python:3.11-slim

WORKDIR /app

# 先拷依赖清单并安装（利用 Docker 层缓存：依赖不变时不重装）
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 拷贝项目代码
COPY agent/ ./agent/
COPY serving/ ./serving/
COPY schemas/ ./schemas/

# 暴露端口
EXPOSE 8080

# 启动 Agent API
CMD ["uvicorn", "serving.agent_api:app", "--host", "0.0.0.0", "--port", "8080"]