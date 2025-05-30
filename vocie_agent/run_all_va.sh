#!/bin/bash

# 1、设置CUDA环境变量
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA 环境变量已设置"

# # 2、启动模型
# # 启动Qwen2.5-14B-Instruct模型
# nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen2.5-14B-Instruct--port 30002 --host 0.0.0.0 --context-length 8192 --max-total-token 10240 --mem-fraction-static 0.45 > logs_file/qwen2.5-14b-instruct.log 2>&1 &
# # 启动Qwen3-0.6B模型
# nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 --context-length 4096 --max-total-token 2048 --mem-fraction-static 0.1 > logs_file/qwen3_0.6b.log 2>&1 &
# # 启动Qwen2.5-3B-Instruct模型
# nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen2.5-3B-Instruct --port 30000 --host 0.0.0.0 --context-length 8192 --max-total-token 9216 --mem-fraction-static 0.3 > logs_file/qwen2.5-3b-instruct.log 2>&1 &
# # 启动Qwen3-14B模型
# nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen3-14B --tool-call-parser qwen25 --port 30004 --host 0.0.0.0 --context-length 8192 --max-total-token 10240 --mem-fraction-static 0.45 > logs_file/qwen3-14b.log 2>&1 &
# # 启动jina-embeddings-v3模型
# nohup uvicorn embedding_server:app --host 0.0.0.0 --port 30003 --reload > logs_file/jina-embed.log 2>&1 &

# # echo "Qwen2.5-14B-Instruct, Qwen2.5-3B-Instruct 和 Qwen3-0.6B 模型已启动"
# echo "jina-embeddings-v3模型已启动"

# 3、启动FastAPI服务
# nohup uvicorn va_client:app --host 0.0.0.0 --port 8002 > logs_file/fastapi.log 2>&1 &
nohup uvicorn va_client:app --host 0.0.0.0 --port 8005 > logs_file/fastapi_new.log 2>&1 &
echo "FastAPI服务已启动"
