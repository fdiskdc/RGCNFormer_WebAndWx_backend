# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 配置阿里云镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir --upgrade pip

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装项目依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# --- 关键修改：使用 Docker 专用文件覆盖标准版 ---
# 这样容器内部的代码在调用 "import tasks" 或 "import config" 时
# 实际上运行的是你修改后的 docker 版本
RUN cp config_docker.py config.py && cp tasks_docker.py tasks.py

# 环境参数移植
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "wsgi:app"]