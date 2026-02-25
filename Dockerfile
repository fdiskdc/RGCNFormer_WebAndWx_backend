# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 1. 替换 apt 软件源为阿里云镜像 (加速系统包下载)
RUN if [ -f /etc/apt/sources.list ]; then \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi && \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources; \
    fi 

# 2. 安装系统基础依赖 (编译 LinearFold 所需)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

# 3. 配置 pip 使用阿里云镜像源
RUN pip install --no-cache-dir --upgrade pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 4. --- 关键步骤：安装 CPU 版 PyTorch 和 PyG 扩展库 ---

# 第一步：安装指定版本的 CPU 版 torch
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 第二步：安装匹配的二进制版扩展库 (scatter, sparse, cluster, spline-conv)
# 这一步非常关键，使用 -f 指定 PyG 官方的 CPU 编译仓库，避开本地编译
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# 第三步：安装其余依赖
COPY requirements.txt .
# 强烈建议：从你的 requirements.txt 中手动删掉以下几行：
# torch, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制代码、编译 LinearFold 等后续操作保持不变
COPY . .
RUN cd LinearFold && make
RUN cp config_docker.py config.py && cp tasks_docker.py tasks.py

# 9. 设置运行环境变量 (防止多进程死锁)
ENV OMP_NUM_THREADS=1 
ENV MKL_NUM_THREADS=1 
ENV PYTHONUNBUFFERED=1 

EXPOSE 8000

# 默认启动 Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]