# RGCNFormer_WebAndWx_backend

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
# RGCNFormer RNA分类后端服务

## 项目简介

RGCNFormer_WebAndWx_backend 是一个基于深度学习的RNA序列分类后端服务，使用图卷积网络（GCN）和类查询注意力机制实现RNA序列的12类多标签分类。该项目支持Web应用和微信小程序两种前端接入方式，并提供丰富的模型可解释性功能。

## 主要特性

- 🧬 **RNA序列分类**：支持12类RNA分类任务
- 🧠 **深度学习模型**：结合多尺度CNN、GCN和Class-Query Attention
- 🔄 **异步处理**：使用Celery实现任务队列和后台处理
- 📦 **缓存机制**：Redis缓存提升响应速度
- 🔍 **模型可解释性**：Integrated Gradients和GCN聚合可视化
- 📱 **微信小程序支持**：完整的用户登录和任务提交接口
- 🐳 **Docker支持**：一键部署，开箱即用
- 🌐 **跨域支持**：CORS配置，方便前端集成

## 技术栈

### 核心框架
- **Flask** - Web应用框架
- **PyTorch** - 深度学习框架
- **PyTorch Geometric** - 图神经网络库
- **Celery** - 分布式任务队列
- **Redis** - 缓存和消息队列

### 关键组件
- **LinearFold** - RNA二级结构预测（编译自C++）
- **Gunicorn** - WSGI HTTP服务器
- **Captum** - PyTorch模型可解释性库

## 项目结构

```
backend/
├── LinearFold/              # RNA二级结构预测工具
│   ├── src/                # C++源代码
│   ├── bin/                # 编译后的二进制文件
│   └── Makefile            # 编译配置
├── json/                   # 配置和数据文件
│   ├── model_graph.json    # 模型计算图
│   └── human.json          # 人类标签映射
├── server.py               # Flask主服务器
├── main_model.py           # 深度学习模型定义
├── tasks.py                # Celery异步任务
├── human.py                # LinearFold接口和工具函数
├── common.py               # 通用常量和配置
├── config.py               # 配置文件
├── Dockerfile              # Docker构建文件
├── docker-compose.yml      # Docker编排文件
└── requirements.txt        # Python依赖
```

## 快速开始

### 环境要求

- Python 3.9+
- Redis服务器
- Docker（推荐）

### 方法1：使用Docker（推荐）

```bash
# 克隆仓库
git clone https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend.git
cd RGCNFormer_WebAndWx_backend

# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 方法2：本地安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装PyTorch Geometric相关包
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install torch-geometric

# 编译LinearFold
cd LinearFold
make
cd ..

# 配置Redis
# 编辑config.py设置Redis连接信息

# 启动Celery Worker
celery -A tasks worker --loglevel=info

# 启动Flask服务器
python server.py
# 或使用Gunicorn
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
```

## API文档

### 基础接口

#### 1. 健康检查
```http
GET /api/health
```

#### 2. 提交预测任务
```http
POST /api/v1/submit-task
Content-Type: application/json

{
  "userId": "user123",
  "rnaSequence": "ACGUACGUACGU...",
  "targetClassId": 0,
  "topK": 10
}
```

#### 3. 获取预测结果
```http
GET /api/v1/results/<job_id>
```

### 微信小程序接口

#### 1. 微信登录
```http
POST /api/v1/wx/login
Content-Type: application/json

{
  "loginCode": "wx_login_code",
  "nickname": "用户昵称",
  "avatarUrl": "头像URL"
}
```

#### 2. 批量提交任务（最多5个序列）
```http
POST /api/v1/wx-submit-task
Content-Type: application/json

{
  "rnaSequence1": "ACGU...",
  "rnaSequence2": "CGUA...",
  "rnaSequence3": "GCAU...",
  "rnaSequence4": "UAUC...",
  "rnaSequence5": "ACGU...",
  "targetClassId": 0,
  "topK": 10
}
```

#### 3. 查询任务进度
```http
GET /api/v1/wx-task-progress/<job_id>
```

### 模型可解释性接口

#### 1. 获取模型架构
```http
GET /api/v1/model-architecture
```

#### 2. 获取模型计算图
```http
GET /api/v1/model-graph
```

#### 3. Integrated Gradients分析
```http
POST /api/v1/integrated-gradients
Content-Type: application/json

{
  "rnaSequence": "ACGUACGUACGU...",
  "targetClassId": 0
}
```

#### 4. GCN聚合可视化
```http
POST /api/v1/visualize-gcn-aggregation
Content-Type: application/json

{
  "rnaSequence": "ACGUACGUACGU...",
  "targetNodeIdx": 10
}
```

## 模型架构

### RNA_ClassQuery_Model

该模型由三个主要组件组成：

1. **ParallelCNNBlock**
   - 多尺度卷积提取局部特征
   - 支持不同核大小的并行卷积分支

2. **GCNBlock**
   - 图卷积网络处理RNA二级结构
   - 支持残差连接和层归一化

3. **ClassQueryHead**
   - 基于注意力机制的类查询头
   - 支持分层分类（12类和4类）

### 配置参数

```json
{
  "model": {
    "cnn_hidden_dim": 64,
    "cnn_kernel_sizes": [1, 3, 5, 7],
    "cnn_dropout": 0.1,
    "gcn_hidden_dim": 128,
    "gcn_out_channels": 128,
    "gcn_num_layers": 3,
    "gcn_dropout": 0.3,
    "num_classes": 12,
    "num_attn_heads": 4,
    "attn_dropout": 0.1,
    "use_simple_pooling": false,
    "use_hierarchical": true,
    "use_layer_norm": true
  }
}
```

## 配置说明

主要配置项位于 `config.py`：

- `FLASK_HOST`: Flask服务器地址
- `FLASK_PORT`: Flask服务器端口
- `FLASK_DEBUG`: 调试模式
- `REDIS_HOST`: Redis服务器地址
- `REDIS_PORT`: Redis服务器端口
- `REDIS_DB`: Redis数据库编号
- `MODEL_DEVICE`: 模型运行设备（cpu/cuda）
- `MODEL_CHECKPOINT_PATH`: 模型权重文件路径
- `MODEL_CONFIG_PATH`: 模型配置文件路径

微信小程序配置：

- `WX_APPID`: 微信小程序AppID
- `WX_SECRET`: 微信小程序AppSecret
- `WX_LOGIN_URL`: 微信登录接口URL

## 部署说明

### 生产环境部署

1. **准备模型文件**
   - 将训练好的模型权重文件放置在指定目录
   - 配置 `MODEL_CHECKPOINT_PATH` 指向权重文件

2. **环境变量配置**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

3. **使用Gunicorn启动**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 --timeout 120 wsgi:app
   ```

4. **使用Nginx反向代理**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker生产部署

```bash
# 构建生产镜像
docker build -t rgcnformer-backend:latest .

# 运行容器
docker run -d \
  --name rgcnformer-backend \
  -p 8000:8000 \
  -v /path/to/model:/app/model \
  -e REDIS_HOST=redis \
  --link redis:redis \
  rgcnformer-backend:latest
```

## 故障排查

### 常见问题

1. **Redis连接失败**
   - 检查Redis服务是否运行
   - 确认配置中的Redis地址和端口正确

2. **模型加载失败**
   - 检查模型权重文件路径是否正确
   - 确认PyTorch和PyG版本匹配

3. **Celery任务不执行**
   - 确认Celery Worker正在运行
   - 检查Celery日志输出

4. **LinearFold编译失败**
   - 确保系统安装了build-essential
   - 检查C++编译器是否可用

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 联系方式

- 项目地址: https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend
- 问题反馈: [GitHub Issues](https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend/issues)

---

<a name="english"></a>
# RGCNFormer RNA Classification Backend Service

## Project Overview

RGCNFormer_WebAndWx_backend is a deep learning-based RNA sequence classification backend service that implements 12-class multi-label classification of RNA sequences using Graph Convolutional Networks (GCN) and Class-Query attention mechanisms. The project supports both Web application and WeChat Mini Program frontends, providing rich model interpretability features.

## Key Features

- 🧬 **RNA Sequence Classification**: Supports 12-class RNA classification tasks
- 🧠 **Deep Learning Model**: Combines multi-scale CNN, GCN, and Class-Query Attention
- 🔄 **Async Processing**: Task queue and background processing using Celery
- 📦 **Caching**: Redis caching for improved response speed
- 🔍 **Model Interpretability**: Integrated Gradients and GCN aggregation visualization
- 📱 **WeChat Mini Program Support**: Complete user login and task submission APIs
- 🐳 **Docker Support**: One-click deployment, ready to use
- 🌐 **CORS Support**: Configured for easy frontend integration

## Tech Stack

### Core Frameworks
- **Flask** - Web application framework
- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library
- **Celery** - Distributed task queue
- **Redis** - Caching and message queue

### Key Components
- **LinearFold** - RNA secondary structure prediction (compiled from C++)
- **Gunicorn** - WSGI HTTP server
- **Captum** - PyTorch model interpretability library

## Project Structure

```
backend/
├── LinearFold/              # RNA secondary structure prediction tool
│   ├── src/                # C++ source code
│   ├── bin/                # Compiled binaries
│   └── Makefile            # Build configuration
├── json/                   # Configuration and data files
│   ├── model_graph.json    # Model computation graph
│   └── human.json          # Human label mapping
├── server.py               # Flask main server
├── main_model.py           # Deep learning model definition
├── tasks.py                # Celery async tasks
├── human.py                # LinearFold interface and utilities
├── common.py               # Common constants and configuration
├── config.py               # Configuration file
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Docker orchestration file
└── requirements.txt        # Python dependencies
```

## Quick Start

### Requirements

- Python 3.9+
- Redis server
- Docker (recommended)

### Method 1: Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend.git
cd RGCNFormer_WebAndWx_backend

# Build and start service
docker-compose up -d

# View logs
docker-compose logs -f
```

### Method 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install torch-geometric

# Compile LinearFold
cd LinearFold
make
cd ..

# Configure Redis
# Edit config.py to set Redis connection info

# Start Celery Worker
celery -A tasks worker --loglevel=info

# Start Flask server
python server.py
# Or use Gunicorn
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 wsgi:app
```

## API Documentation

### Basic Endpoints

#### 1. Health Check
```http
GET /api/health
```

#### 2. Submit Prediction Task
```http
POST /api/v1/submit-task
Content-Type: application/json

{
  "userId": "user123",
  "rnaSequence": "ACGUACGUACGU...",
  "targetClassId": 0,
  "topK": 10
}
```

#### 3. Get Prediction Result
```http
GET /api/v1/results/<job_id>
```

### WeChat Mini Program Endpoints

#### 1. WeChat Login
```http
POST /api/v1/wx/login
Content-Type: application/json

{
  "loginCode": "wx_login_code",
  "nickname": "User Nickname",
  "avatarUrl": "Avatar URL"
}
```

#### 2. Batch Submit Task (up to 5 sequences)
```http
POST /api/v1/wx-submit-task
Content-Type: application/json

{
  "rnaSequence1": "ACGU...",
  "rnaSequence2": "CGUA...",
  "rnaSequence3": "GCAU...",
  "rnaSequence4": "UAUC...",
  "rnaSequence5": "ACGU...",
  "targetClassId": 0,
  "topK": 10
}
```

#### 3. Query Task Progress
```http
GET /api/v1/wx-task-progress/<job_id>
```

### Model Interpretability Endpoints

#### 1. Get Model Architecture
```http
GET /api/v1/model-architecture
```

#### 2. Get Model Computation Graph
```http
GET /api/v1/model-graph
```

#### 3. Integrated Gradients Analysis
```http
POST /api/v1/integrated-gradients
Content-Type: application/json

{
  "rnaSequence": "ACGUACGUACGU...",
  "targetClassId": 0
}
```

#### 4. GCN Aggregation Visualization
```http
POST /api/v1/visualize-gcn-aggregation
Content-Type: application/json

{
  "rnaSequence": "ACGUACGUACGU...",
  "targetNodeIdx": 10
}
```

## Model Architecture

### RNA_ClassQuery_Model

The model consists of three main components:

1. **ParallelCNNBlock**
   - Multi-scale convolution for local feature extraction
   - Supports parallel convolution branches with different kernel sizes

2. **GCNBlock**
   - Graph convolutional network for RNA secondary structure processing
   - Supports residual connections and layer normalization

3. **ClassQueryHead**
   - Attention-based class query head
   - Supports hierarchical classification (12-class and 4-class)

### Configuration Parameters

```json
{
  "model": {
    "cnn_hidden_dim": 64,
    "cnn_kernel_sizes": [1, 3, 5, 7],
    "cnn_dropout": 0.1,
    "gcn_hidden_dim": 128,
    "gcn_out_channels": 128,
    "gcn_num_layers": 3,
    "gcn_dropout": 0.3,
    "num_classes": 12,
    "num_attn_heads": 4,
    "attn_dropout": 0.1,
    "use_simple_pooling": false,
    "use_hierarchical": true,
    "use_layer_norm": true
  }
}
```

## Configuration

Main configuration items are in `config.py`:

- `FLASK_HOST`: Flask server address
- `FLASK_PORT`: Flask server port
- `FLASK_DEBUG`: Debug mode
- `REDIS_HOST`: Redis server address
- `REDIS_PORT`: Redis server port
- `REDIS_DB`: Redis database number
- `MODEL_DEVICE`: Model runtime device (cpu/cuda)
- `MODEL_CHECKPOINT_PATH`: Model weight file path
- `MODEL_CONFIG_PATH`: Model configuration file path

WeChat Mini Program configuration:

- `WX_APPID`: WeChat Mini Program AppID
- `WX_SECRET`: WeChat Mini Program AppSecret
- `WX_LOGIN_URL`: WeChat login API URL

## Deployment Guide

### Production Deployment

1. **Prepare Model Files**
   - Place trained model weights in the specified directory
   - Configure `MODEL_CHECKPOINT_PATH` to point to the weight file

2. **Environment Variables**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

3. **Start with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 --timeout 120 wsgi:app
   ```

4. **Nginx Reverse Proxy**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker Production Deployment

```bash
# Build production image
docker build -t rgcnformer-backend:latest .

# Run container
docker run -d \
  --name rgcnformer-backend \
  -p 8000:8000 \
  -v /path/to/model:/app/model \
  -e REDIS_HOST=redis \
  --link redis:redis \
  rgcnformer-backend:latest
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check if Redis service is running
   - Confirm Redis address and port in configuration are correct

2. **Model Loading Failed**
   - Check if model weight file path is correct
   - Confirm PyTorch and PyG versions match

3. **Celery Tasks Not Executing**
   - Confirm Celery Worker is running
   - Check Celery log output

4. **LinearFold Compilation Failed**
   - Ensure build-essential is installed
   - Check if C++ compiler is available

## Contributing

Issues and Pull Requests are welcome!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Contact

- Project URL: https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend
- Issue Tracker: [GitHub Issues](https://github.com/fdiskdc/RGCNFormer_WebAndWx_backend/issues)