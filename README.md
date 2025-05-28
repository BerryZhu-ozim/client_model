# 双系统 LLM 客户端与部署框架

本文档介绍了智能客服系统（Project 1）和语音机器人系统（Project 2）的客户端选型与部署示例，旨在提供一套基于 FastAPI 的可复用服务模板，用于接入大语言模型（LLM）并结合检索或语音处理组件实现对话能力。

## 环境与依赖

所有实验均在 Ubuntu 系统中、使用 Conda 虚拟环境名为 model_deploy 的环境中创建，采用 Python 3.9.21。硬件平台为配备 NVIDIA A100-SXM4-80GB 的服务器，CUDA 驱动版本为 12.2，（NVIDIA-SMI 535.230.02）。需要确保宿主机已安装 NVIDIA 驱动和 NVIDIA Container Toolkit，可通过在终端运行 `nvidia-smi` 确认当前 GPU 及驱动信息。创建环境并安装依赖命令如下：

```bash
conda create -n model_deploy python=3.9.21
conda activate model_deploy
pip install -r requirements.txt
```

## 安装 SGLang
为了驱动大语言模型（LLM），需要安装 sglang。执行以下命令来安装 sglang 及其所有依赖：
```bash
pip install --upgrade pip
pip install uvicorn
pip install "sglang[all]>=0.4.6.post5"
```

## 安装 FlashInfer（加速推理）：
SGLang 当前使用的是 Torch 2.6，因此需要为 Torch 2.6 安装 FlashInfer。请按照以下步骤安装 FlashInfer：
```bash
# 安装 FlashInfer
curl -sSL https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.3/flashinfer_python-0.2.3+cu124torch2.5-cp38-abi3-linux_x86_64.whl -o /tmp/flashinfer.whl
pip install /tmp/flashinfer.whl
rm /tmp/flashinfer.whl
```

## CUDA 环境变量配置
如果您在安装过程中遇到 OSError: CUDA_HOME environment variable is not set 错误，请设置 CUDA_HOME 环境变量为您的 CUDA 安装路径。您可以通过以下两种方法之一来设置：
```bash
# 方法一：手动设置 CUDA_HOME 环境变量
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
或者，您也可以先安装 FlashInfer，再按上面的说明安装 SGLang。

## 模型服务启动

通过 `sglang.launch_server` 模块启动各型号 LLM 服务实例。以下示例展示了 Qwen3-14B、Qwen3-0.6B 与 Qwen2.5-3B-Instruct 模型的启动方式，特别注意，当我们需要使用模型的agent能力调用工具时，我们需要设定--tool-call-parser参数：

```bash
nohup python -m sglang.launch_server   --model-path Qwen/Qwen3-14B   --tool-call-parser qwen25   --port 30004   --host 0.0.0.0   --context-length 8192   --max-total-token 10240   --mem-fraction-static 0.45   > logs_file/qwen3-14b.log 2>&1 &

nohup python -m sglang.launch_server   --model-path Qwen/Qwen3-0.6B   --port 30001   --host 0.0.0.0   --context-length 8192   --max-total-token 10240   --mem-fraction-static 0.4   > logs_file/qwen3-0.6b.log 2>&1 &

nohup python -m sglang.launch_server   --model-path Qwen/Qwen2.5-3B-Instruct   --port 30000   --host 0.0.0.0   --context-length 8192   --max-total-token 10240   --mem-fraction-static 0.4   > logs_file/qwen2.5-3b-instruct.log 2>&1 &
```

## FastAPI 服务部署

切换到 `voice_agent` 或 `customer_service` 目录后，运行 `uvicorn` 命令启动对应服务：

```bash
uvicorn va_client:app --host 0.0.0.0 --port 8000
uvicorn client_customer_service:app --host 0.0.0.0 --port 8004
```

## 智能客服系统（Project 1）

智能客服系统基于 RAG + LLM + function_call 架构，RAG 数据源为 FAQ 类型知识库，主对话引擎采用 Qwen3-14B，敏感问题检测采用 Qwen3-0.6B。启动后，服务将在 `/chat` 端点接收用户的 JSON 请求，支持动态上传本地 FAQ 知识库（`.xlsx` 或 `.csv` 格式，第一列为 Query、第二列为 Docs），并可通过 `kb_id`、`w_query`、`w_doc`、`top_k` 和 `use_kb` 参数控制检索行为，使用 `session_id` 区分多轮会话。示例请求如下：

```bash
curl -N -X POST 'http://127.0.0.1:8003/chat'   -H 'Content-Type: application/json'   -d '{
    "query": "藏花红一次要泡多少，和他的禁忌有哪些",
    "session_id": "e",
    "kb_id": "default",
    "w_query": 0.9,
    "w_doc": 0.1,
    "top_k": 5,
    "use_kb": true
}'
```

## 语音机器人系统（Project 2）

语音机器人系统流程为拨出→ASR 转 Text→轻量化 LLM 理解并润色→TTS 合成语音，对话模型选用 Qwen2.5-3B-Instruct，敏感问题检测同样采用 Qwen3-0.6B。启动后，可通过以下 JSON 请求示例触发语音机器人对话：
```bash
   curl -N -X POST 'http://127.0.0.1:8002/chat'   -H 'Content-Type: application/json'   -d '{
    "query": " 你是谁创造的",
    "session_id": "c"
  }'
```
或
```json
{
  "query": "Bolehkah anda mengesyorkan saya insurans?",
  "session_id": "a",
  "preset_responses": {
    "insurans": "您可以进入https://implus.com官网，点击insurance按钮就可以看到详细的保险介绍了",
    "足球": "您可以进入https://implus.com官网，点击football按钮就可以看到详细的保险介绍了",
    "film": "您可以进入https://implus.com官网，点击film按钮就可以看到详细的保险介绍了"
  }
}
```

服务启动后，即可按照上述请求示例验证功能，确保系统按预期运行。
