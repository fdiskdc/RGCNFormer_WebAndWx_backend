# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 1. 替换 apt 软件源为阿里云镜像 (加速系统包安装)
# 针对 Debian 11 (Bullseye) 版本的源配置
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 安装系统基础依赖 (对应原脚本中的编译环境需求)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 配置 pip 使用阿里云镜像源并升级 pip
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip

# 4. 安装项目依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制项目代码
COPY . .

# --- 关键修改：使用 Docker 专用文件覆盖标准版 ---
# 确保容器内加载的是适配 Docker 网络 (Redis Host 为 'redis') 的配置
RUN cp config_docker.py config.py && cp tasks_docker.py tasks.py

# 6. 环境参数移植 (对应 start_backend.sh 中的设置)
ENV OMP_NUM_THREADS=1 
ENV MKL_NUM_THREADS=1 
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 默认启动命令
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]