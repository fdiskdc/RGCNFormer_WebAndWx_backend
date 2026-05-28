# DCPRES_WebAndWx_backend

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
# DCPRES RNA分类后端服务

## 项目简介

DCPRES_WebAndWx_backend 是一个基于深度学习的RNA序列分类后端服务，使用图卷积网络（GCN）和类查询注意力机制实现RNA序列的12类多标签分类。该项目支持Web应用和微信小程序两种前端接入方式，并提供丰富的模型可解释性功能和可视化分析。

## 主要特性

- 🧬 **RNA序列分类**：支持12类RNA分类任务
- 🧠 **深度学习模型**：结合多尺度CNN、GCN和Class-Query Attention
- 🔄 **异步处理**：使用Celery实现任务队列和后台处理
- 📦 **缓存机制**：Redis缓存提升响应速度
- 🔍 **模型可解释性**：Integrated Gradients和GCN聚合可视化
- 📊 **模型对比分析**：多模型性能对比（DCPRES, ModX, MultiRM）
- 🗺️ **UMAP可视化**：RNA序列嵌入的降维可视化
- 📈 **分类性能热图**：12类×9指标的热力图展示
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
git clone https://github.com/fdiskdc/DCPRES_WebAndWx_backend.git
cd DCPRES_WebAndWx_backend

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

#### 5. 获取示例序列
```http
GET /api/v1/sample-sequence
```
返回工作区输入块的随机示例序列。

### 模型对比与分析接口

#### 1. 模型性能对比
```http
GET /api/v1/model-comparison
```
返回DCPRES、ModX、MultiRM三个模型的平均性能指标（Acc, AUC, AUPRC, Precision, Recall, F1, MCC, Sn, Sp）。

#### 2. DCPRES分类热图
```http
GET /api/v1/rgcnformer-classification-heatmap
```
返回DCPRES模型12类×9指标的热力图数据。

#### 3. DCPRES定位性能
```http
GET /api/v1/rgcnformer-localization
```
返回DCPRES定位性能数据（12类×7个Top-K值），用于甜甜圈图和统计表。

#### 4. 定位模型对比
```http
GET /api/v1/rgcnformer-loc-comparison
```
返回DCPRES、ModX、MultiRM三个模型的定位性能对比（气泡图数据）。

#### 5. UMAP可视化数据
```http
GET /api/v1/umap-data?n=1000
```
返回预计算的UMAP坐标和元数据，支持通过`n`参数进行下采样。

## 模型架构

### DCPRES_Model

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
- `MODEL_COMPARISON_CSV_DIR`: Model comparison CSV files directory
- `MODEL_COMPARISON_FILES`: Model comparison file mapping (ModX, MultiRM, DCPRES)
- `UMAP_DATA_PATH`: UMAP visualization data file path

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
docker build -t dcpres-backend:latest .

# Run container
docker run -d \
  --name dcpres-backend \
  -p 8000:8000 \
  -v /path/to/model:/app/model \
  -e REDIS_HOST=redis \
  --link redis:redis \
  dcpres-backend:latest
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

- Project URL: https://github.com/fdiskdc/DCPRES_WebAndWx_backend
- Issue Tracker: [GitHub Issues](https://github.com/fdiskdc/DCPRES_WebAndWx_backend/issues)