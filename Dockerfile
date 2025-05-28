# 使用 NVIDIA 提供的 CUDA 基础镜像，适用于 GPU 加速
FROM nvidia/cuda:12.9-base-ubuntu22.04

# 设置环境变量，避免交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y \
    python3.9 \
    python3-pip \
    wget \
    git \
    curl \
    build-essential \
    && apt-get clean

# 设置 Python 版本
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# 安装 Conda
RUN curl -sSL https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh -o /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh && \
    ln -s /opt/conda/bin/conda /usr/bin/conda

# 创建并激活 Conda 环境
RUN conda create -n model_deploy python=3.9.21 && \
    echo "conda activate model_deploy" >> ~/.bashrc

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 安装 FlashInfer（用于加速推理）
RUN curl -sSL https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.3/flashinfer_python-0.2.3+cu124torch2.5-cp38-abi3-linux_x86_64.whl -o /tmp/flashinfer.whl && \
    pip install /tmp/flashinfer.whl && \
    rm /tmp/flashinfer.whl

# 安装 SGLang 及其依赖
RUN pip install "sglang[all]>=0.4.6.post5" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 下载模型
RUN python download_model.py

# 创建日志目录
RUN mkdir -p logs_file

# 启动模型服务
CMD nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen3-0.6B --port 30001 --host 0.0.0.0 --context-length 4096 --max-total-token 2048 --mem-fraction-static 0.1 > logs_file/qwen3_0.6b.log 2>&1 & \
    nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen2.5-3B-Instruct --port 30000 --host 0.0.0.0 --context-length 8192 --max-total-token 9216 --mem-fraction-static 0.3 > logs_file/qwen2.5-3b-instruct.log 2>&1 & \
    nohup python -m sglang.launch_server --model-path local_models/Qwen/Qwen3-14B --tool-call-parser qwen25 --port 30004 --host 0.0.0.0 --context-length 8192 --max-total-token 10240 --mem-fraction-static 0.45 > logs_file/qwen3-14b.log 2>&1 & \
    nohup uvicorn embedding_server:app --host 0.0.0.0 --port 30003 --reload > logs_file/jina-embed.log 2>&1 & \
    nohup uvicorn voice_agent.va_client:app --host 0.0.0.0 --port 8002 > logs_file/fastapi_va.log 2>&1 & \
    nohup uvicorn customer_service.client_customer_service:app --host 0.0.0.0 --port 8003 > logs_file/fastapi_cs.log 2>&1 & \

