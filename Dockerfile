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

# 4. --- 关键步骤：安装 CPU 版 PyTorch 相关组件 ---
# 首先安装 CPU 版 torch 和 torchvision (如果需要)
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html 

# 接着安装匹配的二进制版 torch-scatter (无需本地编译)
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cpu.html 

# 5. 安装剩余 Python 项目依赖
COPY requirements.txt .
# 注意：请确保你的 requirements.txt 中删除了 torch 和 torch-scatter，避免重复安装或版本冲突
RUN pip install --no-cache-dir -r requirements.txt 

# 6. 复制项目代码
COPY . . 

# 7. 编译 LinearFold
RUN cd LinearFold && make 

# 8. 处理 Docker 专用配置
RUN cp config_docker.py config.py && cp tasks_docker.py tasks.py 

# 9. 设置运行环境变量 (防止多进程死锁)
ENV OMP_NUM_THREADS=1 
ENV MKL_NUM_THREADS=1 
ENV PYTHONUNBUFFERED=1 

EXPOSE 8000

# 默认启动 Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]