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

# 3. 配置 pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 4. --- 关键修正部分 ---

# 第一步：安装 CPU 版 PyTorch
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 第二步：安装 PyG 的四个底层二进制扩展库 (CPU版)
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# 第三步：【显式安装】torch-geometric
# 放在安装 requirements.txt 之前，确保它能认到上面安装好的底层库
RUN pip install --no-cache-dir torch-geometric

# 第四步：安装其他剩余依赖
COPY requirements.txt .
# 提醒：务必确保 requirements.txt 里没有上述已安装的库
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制、编译 LinearFold、覆盖配置
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