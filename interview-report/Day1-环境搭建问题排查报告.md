# Paper2Code Day 1：vLLM 服务搭建中的问题排查报告

> 面试用 · 技术问题复盘
> 项目：Paper2Code（论文复现 Agent 系统）
> 阶段：第一周 Day 1 —— 在本地 GPU 机器上把 vLLM 模型服务跑通，打通「客户端 → OpenAI 兼容接口 → 本地模型」最小链路。
> 技术栈：vLLM + Qwen2.5-3B-Instruct + OpenAI SDK，单机多卡（RTX 3090 24G）。

---

## 一句话总结（面试可直接讲）

> 在本地用 vLLM 部署模型时，连续踩了**依赖安装、端口占用、多卡显存争用、模型下载**四个坑。我的处理思路是统一的：**先用最小命令复现 → 看完整报错/现状 → 定位根因 → 用最小代价修复并固化到脚本里**，最终把一条「一键启动 + 一键验证」的可复现链路沉淀下来，方便后面反复调试。

下面四个问题相互独立，面试时可以挑 1~2 个展开讲。每个都按 **现象 → 排查 → 根因 → 解决 → 收获** 组织。

---

## 问题一：`pip install vllm` 卡在 `pyairports` 依赖，装不上

### 现象
执行 `pip install vllm` 时，绝大部分包都装好了，却卡在一个很冷门的传递依赖 `pyairports` 上，反复超时 / 拉取失败，导致整个 vLLM 安装失败。

### 排查过程
1. 先确认不是 vllm 本身的问题，而是某个依赖拉不下来——从报错里锁定是 `pyairports`。
2. 直接查 PyPI 上这个包到底有没有、版本是什么：
   ```bash
   curl -s -m 15 https://pypi.org/pypi/pyairports/json
   curl -s -m 15 https://pypi.org/pypi/pyairports/0.0.1/json
   ```
3. 拿到 `files.pythonhosted.org` 上的真实下载地址，手动下载并校验文件大小（确认不是 0 字节的坏包）：
   ```bash
   curl -s -m 30 -o pyairports-0.0.1.tar.gz "https://files.pythonhosted.org/.../pyairports-0.0.1.tar.gz"
   stat -c%s pyairports-0.0.1.tar.gz   # 确认体积正常
   tar tzf pyairports-0.0.1.tar.gz     # 确认包结构完整
   ```

### 根因
不是包本身有问题，而是**网络到 PyPI 的链路不稳定**，pip 在解析这个传递依赖时反复超时。冷门小包没有被国内常用镜像很好地覆盖，所以默认源容易失败。

### 解决
- 短期：手动把包下到本地、校验完整性后离线安装，让 vLLM 安装链路先走通。
- 固化：把 pip 源切到稳定的国内镜像，避免后续每个依赖都赌一次网络。
- 同时在 `requirements.txt` 里写清**安装顺序**（先单独 `pip install vllm` 让它带上配套 torch，再装其余），从根上减少这类卡顿。

### 收获
遇到「装不上」先别盲目重试。**把 pip 的黑盒拆开**：直接查 PyPI 的 JSON 接口确认包存在性和真实下载地址，再手动验证文件完整性，能快速区分到底是「源/网络问题」还是「包本身损坏」，两者的修法完全不同。

---

## 问题二：端口被占用，服务起不来 / 客户端连不上

### 现象
按 README 把服务起在默认的 `8000` 端口，但要么启动报端口占用，要么客户端连过去拿到的是别的服务的响应——这台机器是共享开发机，`8000` 早被别的进程占了。

### 排查过程
直接探活，看每个端口上到底有没有、有的话是不是我的 vLLM：
```bash
curl -s -m 5 http://localhost:8000/v1/models   # 被别人占了
curl -s -m 5 http://localhost:8001/v1/models   # 换一个
```
通过 `/v1/models` 返回的模型名判断这个端口上的服务是不是自己启的那个。

### 根因
共享机器上端口是公共资源，默认端口经常冲突。

### 解决
- 把 vLLM 服务迁到空闲端口（`--port 8001`），并保证**三处端口一致**：
  - 启动脚本 `start_vllm.sh` 的 `--port`
  - 验证脚本 `test_vllm.py` 的 `base_url`
  - 客户端封装 `serving/llm_client.py` 的 `BASE_URL`
- 把端口作为**唯一可信来源**集中在一处，避免「改了启动端口忘了改客户端」这种低级不一致。

### 收获（也是一个诚实的反思）
排查时发现 `start_vllm.sh` 里 `echo` 提示的还是 `http://localhost:8000`，但实际 `--port 8001`——**提示信息和真实行为脱节**，这种不一致最坑人，会把排查带偏。教训是：配置类的值要么集中成变量、要么就别在注释/提示里写死第二份，否则迟早不同步。

---

## 问题三：多卡机器上 vLLM 启动 OOM / 抢到了别人正在用的卡

### 现象
机器有多张卡，vLLM 默认抓 `0` 号卡，但 `0` 号卡已经被别的任务占了大半显存，启动直接 OOM 或者和别人抢资源。

### 排查过程
先看清楚每张卡的真实占用，再决定用哪张：
```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv
```
确认 `0` 号卡被占、`1` 号卡空闲。

### 根因
两层原因叠加：
1. **没指定卡**：vLLM 默认用 `0` 号，恰好是被占的那张。
2. **显存预占比例**：vLLM 启动时按 `--gpu-memory-utilization` 一次性预占显存做 KV cache，比例给太高会和系统/其他进程的余量冲突。

### 解决
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.85 \   # 留 15% 余量给系统，避免边界 OOM
  --max-model-len 8192 \            # 小模型上下文设小，省显存
  --port 8001
```
- `CUDA_VISIBLE_DEVICES=1` 显式锁定空闲卡，不和别人抢。
- `--gpu-memory-utilization 0.85` 留余量，是稳定性和吞吐之间的折中。
- 调试中起错/起重的进程，用 `pkill -f "port 8002"` 精准清掉，不误伤别人。

### 收获
- 第一周特意**用 3B 小模型**而不是直接上 31B，就是为了让「显存毫无压力、启动快」，把精力集中在打通链路上；把「31B + 量化」这种显存硬骨头留到第二周。**先跑通、再优化**。
- 理解了 vLLM 的显存模型：它是**预占式**的，`gpu-memory-utilization` 和 `max-model-len` 共同决定 KV cache 能开多大，这俩参数是后续吞吐调优的关键旋钮。

---

## 问题四：HuggingFace 模型下载极慢 / 中断

### 现象
首次启动 vLLM 要从 HuggingFace 拉 Qwen2.5-3B 权重，直连速度极慢甚至中断。

### 根因
直连 huggingface.co 在当前网络环境下不稳定。

### 解决
启动前设置镜像端点，让 vLLM/transformers 全程走镜像下载：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
并把它写进 `start_vllm.sh`，保证每次启动都生效，不依赖个人 shell 环境。

### 收获
环境相关的「隐性前提」（镜像、端口、用哪张卡）一定要**固化进脚本**，而不是停留在「我本地记得这么设」。这是可复现性的底线——也是后面能把整套流程写进 README「快速开始」的前提。

---

## 整体复盘：我的排查方法论（面试加分项）

把上面四个问题抽象出来，其实是同一套打法：

1. **最小复现**：用一条命令（`curl /v1/models`、`nvidia-smi --query`、`pip` 单包）把问题暴露出来，不在大流程里猜。
2. **看现状而非看猜测**：报错信息、端口探活结果、显存占用，都是**事实**；先拿到事实再下结论。
3. **拆黑盒**：pip 装不上就直接查 PyPI 接口；服务连不上就直接打 `/v1/models`。绕过封装去看底层，能快速区分根因层次。
4. **最小代价修复 + 固化**：修好之后一定**写回脚本/配置**（端口集中、镜像写进启动脚本、安装顺序写进 requirements），让问题不再复发，也让别人能一键复现。

最终产出：一套「`check_env.py` 自检 → `start_vllm.sh` 一键启动 → `test_vllm.py` 一键验证」的可复现链路，Day 1 验收（模型正常回复）通过。

---

## 可能被追问的问题（提前准备）

- **Q：为什么不用 Ollama / 直接 transformers，而用 vLLM？**
  A：vLLM 有 PagedAttention 和连续批处理，吞吐高，而且暴露 OpenAI 兼容接口，后面接 LangGraph / LangChain 生态零成本；同样的客户端代码，第二周把 3B 换成 31B 只改 `llm_client.py` 一处即可。
- **Q：`gpu-memory-utilization` 设 0.85 是怎么定的？**
  A：是稳定性和 KV cache 容量的折中。留 15% 给系统和显存碎片，避免边界 OOM；如果要追吞吐可以往上调，但要配合 `max-model-len` 一起看。
- **Q：端口/卡这些为什么不用配置文件管理？**
  A：第一周追求最小闭环，先用脚本里的变量集中管理；工程化（配置文件 / 环境变量 / FastAPI + Docker）是第三周的事，避免过早抽象。
