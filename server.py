from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import redis
import hashlib
import numpy as np
import torch
import requests
from torch_geometric.data import Batch
from captum.attr import IntegratedGradients
from celery.result import AsyncResult

# Import local modules
from main_model import RNA_ClassQuery_Model
from human import run_linearfold, build_edge_index_from_structure
from common import INDEX_TO_NUCLEOTIDE
from tasks import celery_app, run_prediction_task
from config import config, get_logger

# Initialize logger
logger = get_logger('server')

# ============================================================================
# Redis Connection (Shared Storage for Multi-Worker Setup)
# ============================================================================
try:
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info(f"Connected to Redis successfully at {config.REDIS_HOST}:{config.REDIS_PORT}")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    logger.warning("Falling back to in-memory storage (not suitable for multi-worker gunicorn)")
    redis_client = None


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence string to one-hot encoding.

    Args:
        sequence: RNA sequence string (A, C, G, U)

    Returns:
        One-hot encoded array of shape (len(sequence), 4)
    """
    one_hot_mapping = {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'T': [0., 0., 0., 1.],  # Treat T as U
        'N': [0., 0., 0., 0.]
    }

    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in one_hot_mapping:
            one_hot[i] = one_hot_mapping[nucleotide]
        else:
            one_hot[i] = [0., 0., 0., 0.]  # Unknown nucleotide

    return one_hot

# 初始化 Flask app
app = Flask(__name__)
# 设置 CORS，允许来自前端（例如 http://localhost:5173）的跨域请求
CORS(app)

# ============================================================================
# Global Model Loading (Load once at startup)
# ============================================================================

logger.info("Loading model and configuration...")

# Load configuration
with open(config.MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

model_cfg = model_config['model']

# Initialize model
device = config.MODEL_DEVICE
logger.info(f"Using device: {device}")

model = RNA_ClassQuery_Model(
    cnn_hidden_dim=model_cfg['cnn_hidden_dim'],
    cnn_kernel_sizes=tuple(model_cfg['cnn_kernel_sizes']),
    cnn_dropout=model_cfg['cnn_dropout'],
    gcn_hidden_dim=model_cfg['gcn_hidden_dim'],
    gcn_out_channels=model_cfg['gcn_out_channels'],
    gcn_num_layers=model_cfg['gcn_num_layers'],
    gcn_dropout=model_cfg['gcn_dropout'],
    num_classes=model_cfg['num_classes'],
    num_attn_heads=model_cfg['num_attn_heads'],
    attn_dropout=model_cfg['attn_dropout'],
    use_simple_pooling=model_cfg['use_simple_pooling'],
    use_hierarchical=model_cfg['use_hierarchical'],
    use_layer_norm=model_cfg['use_layer_norm']
)

# Load model weights
checkpoint_path = config.MODEL_CHECKPOINT_PATH
checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

logger.info(f"Model loaded successfully from {checkpoint_path}")
logger.info("Model is ready for predictions!")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "checkpoint": checkpoint_path
    })

# ============================================================================
# WeChat Mini Program Login Endpoint
# ============================================================================

@app.route('/api/v1/wx/login', methods=['POST'])
def wx_login():
    """
    WeChat Mini Program login endpoint.
    Expects JSON body with 'loginCode' key.
    Returns openid after exchanging code with WeChat API.
    """
    # Get data from request
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    login_code = data.get('loginCode')
    
    if not login_code:
        return jsonify({"error": "loginCode is required"}), 400
    
    # Check if WeChat app credentials are configured
    if not config.WX_APPID or not config.WX_SECRET:
        logger.error("WeChat app credentials not configured")
        return jsonify({
            "error": "WeChat app credentials not configured",
            "detail": "Please set WX_APPID and WX_SECRET environment variables"
        }), 500
    
    logger.info(f"Processing WeChat login request for code: {login_code[:10]}...")
    
    try:
        # Call WeChat API to exchange code for openid and session_key
        params = {
            'appid': config.WX_APPID,
            'secret': config.WX_SECRET,
            'js_code': login_code,
            'grant_type': 'authorization_code'
        }
        
        response = requests.get(config.WX_LOGIN_URL, params=params, timeout=10)
        response_data = response.json()
        
        # Check if WeChat API returned an error
        if 'errcode' in response_data:
            error_msg = response_data.get('errmsg', 'Unknown WeChat API error')
            logger.error(f"WeChat API error: {response_data.get('errcode')} - {error_msg}")
            return jsonify({
                "error": f"WeChat API error: {response_data.get('errcode')}",
                "detail": error_msg
            }), 400
        
        # Extract openid and session_key
        openid = response_data.get('openid')
        session_key = response_data.get('session_key')
        
        if not openid:
            logger.error("WeChat API did not return openid")
            return jsonify({
                "error": "Failed to get openid from WeChat API",
                "detail": response_data
            }), 500
        
        # Check if user already exists in Redis
        user = {
            'openid': openid,
            'session_key': session_key,
            'nickname': None,
            'avatarUrl': None
        }
        
        if redis_client:
            try:
                user_key = f"wx_user:{openid}"
                user_data = redis_client.get(user_key)
                
                if user_data:
                    existing_user = json.loads(user_data)
                    logger.info(f"Existing user found: {openid}")
                    # Update session_key for security and preserve nickname/avatar
                    user['session_key'] = session_key
                    user['nickname'] = existing_user.get('nickname')
                    user['avatarUrl'] = existing_user.get('avatarUrl')
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                else:
                    # Store new user data in Redis with 30 days TTL
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                    logger.info(f"New user created: {openid}")
            except Exception as e:
                logger.error(f"Redis error during user storage: {e}")
                # Continue with minimal user object even if Redis fails
        
        # Prepare user info for response (exclude session_key for security)
        user_info = {
            'openid': user['openid'],
            'nickname': user.get('nickname'),
            'avatarUrl': user.get('avatarUrl')
        }
        
        # Return success response with user info
        return jsonify({
            "code": 0,
            "openid": openid,
            "data": user_info,
            "message": "Login successful"
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("WeChat API request timeout")
        return jsonify({
            "error": "WeChat API request timeout",
            "detail": "Failed to connect to WeChat server"
        }), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"WeChat API request error: {e}")
        return jsonify({
            "error": "Failed to call WeChat API",
            "detail": str(e)
        }), 500
    except Exception as e:
        import traceback
        error_msg = f"WeChat login error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500

# ============================================================================
# Prediction Endpoint
# ============================================================================

@app.route('/api/v1/submit-task', methods=['POST'])
def submit_task():
    """
    Submit a prediction task for asynchronous processing.
    Returns immediately with a job_id and 'pending' status.
    The actual prediction runs in the background via Celery.
    """
    # Get data from request JSON body
    data = request.get_json()
    user_id = ""
    original_sequence = ""
    target_class_id = None
    top_k = None
    if data:
        user_id = data.get('userId', '')
        original_sequence = data.get('rnaSequence', '')
        target_class_id = data.get('targetClassId')  # Optional: specific class to visualize
        top_k = data.get('topK')  # Optional: number of top sites to display
        # Print logs for debugging
        logger.info(f"Received user_id: {user_id}, target_class_id: {target_class_id}, top_k: {top_k}")
        logger.info(f"Received sequence from frontend: {original_sequence[:50]}... (length: {len(original_sequence)})")
        if target_class_id is not None:
            logger.info(f"Target class ID: {target_class_id}")
        if top_k is not None:
            logger.info(f"Top-K value: {top_k}")

    # Validate sequence
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400

    # Step 1: Generate SHA256 hash of the RNA sequence as job_id
    job_id = hashlib.sha256(original_sequence.encode('utf-8')).hexdigest()
    logger.info(f"Generated job_id (SHA256 hash): {job_id}")

    # Step 2: Check Redis cache for existing result
    if redis_client:
        try:
            cached_result = redis_client.get(f"task:{job_id}")
            if cached_result:
                logger.info(f"Cache HIT for job_id: {job_id}")
                # Deserialize and return cached result immediately
                result = json.loads(cached_result)
                # Update jobId to match the hash
                result["jobId"] = job_id
                result["status"] = "completed"
                return jsonify(result), 200
            else:
                logger.info(f"Cache MISS for job_id: {job_id}")
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            # Continue with task submission if Redis fails
    else:
        logger.warning("Redis not available, skipping cache check")

    # Step 3: Submit Celery task for background processing
    try:
        # Submit the prediction task to Celery
        # Use the sequence hash as the task_id for consistency
        run_prediction_task.apply_async(
            args=[original_sequence, target_class_id, top_k],
            task_id=job_id
        )

        logger.info(f"Task {job_id} submitted to Celery for background processing")

        # Return immediately with 202 Accepted
        response = {
            "jobId": job_id,
            "status": "pending"
        }
        return jsonify(response), 202

    except Exception as e:
        import traceback
        error_msg = f"Failed to submit task: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500

# ============================================================================
# Results Retrieval Endpoint
# ============================================================================

@app.route('/api/v1/results/<job_id>', methods=['GET'])
def get_result(job_id):
    """
    Retrieve the result of a previously submitted task by job_id.
    First checks Redis cache for completed results.
    If not found, queries Celery result backend for task status.
    Returns:
    - 200 with full result if task completed and cached
    - 200 with {"status": "processing"} if task is still running
    - 200 with {"status": "failed", "error": "..."} if task failed
    - 404 if task not found
    """
    try:
        # Step 1: Check Redis cache first (fastest path)
        if redis_client:
            result_json = redis_client.get(f"task:{job_id}")
            if result_json:
                result = json.loads(result_json)
                return jsonify(result), 200
            logger.info(f"Cache MISS for job_id: {job_id} in results endpoint")

        # Step 2: Check Celery task status
        task = AsyncResult(job_id, app=celery_app)

        if task.state == 'PENDING':
            # Task is waiting to be processed or currently processing
            logger.info(f"Task {job_id} status: PENDING (processing)")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        elif task.state == 'STARTED':
            # Task is currently being processed (if task_track_started=True)
            logger.info(f"Task {job_id} status: STARTED (processing)")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        elif task.state == 'SUCCESS':
            # Task completed successfully - result should be in Redis by now
            # If we reach here, it means the result wasn't in Redis but task is done
            # This can happen if Redis caching failed in the task
            logger.info(f"Task {job_id} status: SUCCESS but not in cache")
            result = task.result
            if isinstance(result, dict):
                # Try to cache it now for future requests
                if redis_client:
                    try:
                        result_json = json.dumps(result, ensure_ascii=False)
                        redis_client.setex(f"task:{job_id}", config.REDIS_CACHE_TTL, result_json)
                    except Exception as e:
                        logger.error(f"Failed to cache result in Redis: {e}")
                return jsonify(result), 200
            else:
                return jsonify({"error": "Invalid result format"}), 500
        elif task.state == 'FAILURE':
            # Task failed with an exception
            logger.info(f"Task {job_id} status: FAILURE")
            error_info = task.info

            # Extract structured error information
            error_response = {
                "jobId": job_id,
                "status": "failed"
            }

            if isinstance(error_info, dict):
                # Check for structured error info from TaskError
                if 'error_info' in error_info:
                    structured_error = error_info['error_info']
                    error_response["error"] = structured_error.get('error_message', 'Unknown error')
                    error_response["errorType"] = structured_error.get('error_type', 'Unknown')
                    error_response["step"] = structured_error.get('step', 'unknown')
                else:
                    # Fallback for standard Celery error format
                    error_response["error"] = error_info.get('message', str(error_info))
                    error_response["errorType"] = error_info.get('error_type', type(error_info).__name__)
            else:
                # String error format
                error_response["error"] = str(error_info)
                error_response["errorType"] = "Unknown"

            return jsonify(error_response), 200
        elif task.state == 'RETRY':
            # Task is being retried
            logger.info(f"Task {job_id} status: RETRY")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        else:
            # Unknown state
            logger.info(f"Task {job_id} status: {task.state}")
            return jsonify({
                "jobId": job_id,
                "status": "unknown",
                "state": task.state
            }), 200

    except Exception as e:
        logger.error(f"Error retrieving job result: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to retrieve result"}), 500

# ============================================================================
# Integrated Gradients Endpoint
# ============================================================================

def extract_module_info(module, module_name=""):
    """
    Recursively extract information from a PyTorch nn.Module hierarchy.
    
    Args:
        module: PyTorch nn.Module object
        module_name: Name of the module (for display)
    
    Returns:
        Dictionary containing module structure information
    """
    # Get class name
    class_name = module.__class__.__name__
    
    # Build result dictionary
    result = {
        "name": module_name,
        "class": class_name,
        "details": "",
        "children": []
    }
    
    # Extract relevant details based on module type
    if class_name == "ParallelCNNBlock":
        details = f"Kernels: {module.kernel_sizes}, Hidden: {module.hidden_dim}"
        result["details"] = details
        
        # Add convolution branches as children
        for i, conv in enumerate(module.conv_branches):
            conv_info = {
                "name": f"conv_{i}",
                "class": "Conv1d",
                "details": f"kernel_size={module.kernel_sizes[i]}, out_channels={module.hidden_dim}",
                "children": []
            }
            result["children"].append(conv_info)
            
    elif class_name == "GCNBlock":
        details = f"Layers: {module.num_layers}, Hidden: {module.hidden_dim}, Out: {module.out_channels}"
        result["details"] = details
        
        # Add GCN layers as children
        for i, gcn_layer in enumerate(module.gcn_layers):
            gcn_info = {
                "name": f"gcn_layer_{i}",
                "class": "GCNConv",
                "details": f"layer_{i}",
                "children": []
            }
            result["children"].append(gcn_info)
            
    elif class_name == "ClassQueryHead":
        details = f"Classes: {module.num_classes}, Heads: {module.num_heads if hasattr(module, 'num_heads') else 'N/A'}"
        result["details"] = details
        
        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)
            
    elif class_name == "ClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}"
        result["details"] = details
        
        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)
            
    elif class_name == "HierarchicalClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}, Groups: {module.num_groups}"
        result["details"] = details
        
        # Add group queries as children
        for i, group_name in enumerate(module.group_names):
            group_info = {
                "name": f"group_query_{i}",
                "class": "GroupQuery",
                "details": f"{group_name} group",
                "children": []
            }
            
            # Add derived class queries
            if group_name in module.group_to_class_indices:
                class_indices = module.group_to_class_indices[group_name]
                for j, class_idx in enumerate(class_indices):
                    class_info = {
                        "name": f"class_query_{class_idx}",
                        "class": "DerivedClassQuery",
                        "details": f"class_{class_idx}",
                        "children": []
                    }
                    group_info["children"].append(class_info)
            
            result["children"].append(group_info)
    
    # Recursively process named children
    for name, child in module.named_children():
        if not child.__class__.__name__ in ["Sequential", "ModuleList", "ModuleDict"]:
            child_info = extract_module_info(child, name)
            result["children"].append(child_info)
    
    return result


@app.route('/api/v1/model-architecture', methods=['GET'])
def get_model_architecture():
    """
    Get the hierarchical structure of the RNA_ClassQuery_Model.
    Returns a JSON representation of the model's architecture.
    """
    try:
        logger.info("Extracting model architecture...")
        
        # Extract model structure
        model_info = extract_module_info(model, "RNA_ClassQuery_Model")
        
        logger.info("Model architecture extracted successfully")
        
        return jsonify(model_info), 200
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to extract model architecture: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/model-graph', methods=['GET'])
def get_model_graph():
    """
    Get the ONNX model computation graph (nodes and edges).
    Returns the model graph data from the JSON file.
    """
    try:
        logger.info("Loading model graph data...")
        
        # Read model graph from JSON file
        model_graph_path = 'json/model_graph.json'
        
        with open(model_graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        logger.info(f"Model graph loaded successfully: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
        
        return jsonify(graph_data), 200
        
    except FileNotFoundError:
        logger.error(f"Model graph file not found: {model_graph_path}")
        return jsonify({
            "error": "Model graph file not found",
            "detail": f"The file {model_graph_path} does not exist. Please generate the model graph first.",
            "type": "FileNotFoundError"
        }), 404
    except Exception as e:
        import traceback
        error_msg = f"Failed to load model graph: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/integrated-gradients', methods=['POST'])
def integrated_gradients():
    """
    Compute Integrated Gradients attributions for RNA sequence prediction.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_class_id = data.get('targetClassId')
    
    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400
    
    if target_class_id is None or not (0 <= target_class_id < 12):
        return jsonify({"error": "Invalid targetClassId. Must be between 0 and 11"}), 400
    
    logger.info(f"Integrated Gradients: sequence length={len(original_sequence)}, target_class_id={target_class_id}")
    
    # Store original sequence for response
    sequence = original_sequence
    
    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)
    
    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0
    
    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
    
    try:
        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]
        
        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)
        
        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]
        
        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)
        
        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)
        
        # Step 5: Compute Integrated Gradients
        # Check if model is hierarchical
        if model_cfg['use_hierarchical']:
            # For hierarchical model, wrap the model to extract 12-class logits
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                # output is a tuple: (logits_12class, logits_4class, attn_weights)
                logits_12class = output[0]  # Extract 12-class logits
                return logits_12class
        else:
            # For non-hierarchical model, use model directly
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                return output
        
        # Instantiate IntegratedGradients
        ig = IntegratedGradients(forward_func)
        
        # Compute attributions
        with torch.enable_grad():
            # Baseline: zero tensor of same shape
            baseline = torch.zeros_like(x)
            
            # Compute attributions
            attributions = ig.attribute(
                x.unsqueeze(0),  # Add batch dimension
                baselines=baseline.unsqueeze(0),
                target=target_class_id,
                additional_forward_args=(edge_index, batch),
                internal_batch_size=1
            )
        
        # Step 6: Process attributions
        attributions = attributions.squeeze(0)  # Remove batch dimension [1001, 4]
        
        # Sum attributions across the one-hot encoding dimension to get per-nucleotide score
        node_attributions = attributions.sum(dim=1).cpu().numpy()  # [1001]
        
        # Step 7: Build GCN graph data with attribution scores
        edge_index_np = edge_index.cpu().numpy()
        edges = []
        
        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)
        
        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            
            # Only process edges within valid range
            if not (0 <= source < len(sequence) and 0 <= target < len(sequence)):
                continue
            
            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming
            
            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue
            
            # Only keep one direction (source < target) to avoid duplicates
            if source >= target:
                continue
            
            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })
        
        # Create nodes with attribution scores
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            # Map original index to model index
            model_index = i + left_padding - left_trimming
            attribution_score = float(node_attributions[model_index]) if 0 <= model_index < len(node_attributions) else 0.0
            
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {
                    "index": i,
                    "type": nuc,
                    "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}",
                    "attributionScore": attribution_score
                }
            })
        
        # Create a set of valid node IDs for filtering edges
        valid_node_ids = {node["id"] for node in nodes}
        
        # Filter edges to only include those that reference valid nodes
        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]
        
        logger.info(f"Integrated Gradients: 节点数={len(nodes)}, 有效边数={len(valid_edges)}")
        
        # Return response
        return jsonify({
            "nodes": nodes,
            "edges": valid_edges,
            "targetClassId": target_class_id
        }), 200
        
    except Exception as e:
        import traceback
        error_msg = f"Integrated Gradients error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/visualize-gcn-aggregation', methods=['POST'])
def visualize_gcn_aggregation():
    """
    Visualize GCN message passing for a specific target node.
    Returns aggregation details showing message strengths from neighbors at each GCN layer.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_node_idx = data.get('targetNodeIdx')
    
    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400
    
    if target_node_idx is None or not isinstance(target_node_idx, int):
        return jsonify({"error": "Invalid targetNodeIdx. Must be an integer"}), 400
    
    logger.info(f"GCN Aggregation Viz: sequence length={len(original_sequence)}, target_node_idx={target_node_idx}")
    
    # Store original sequence for response
    sequence = original_sequence
    
    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)
    
    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0
    
    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
    
    # Map original target node index to model coordinates
    model_target_idx = target_node_idx + left_padding - left_trimming
    
    # Validate model target index
    if not (0 <= model_target_idx < len(sequence)):
        return jsonify({"error": f"Target node index out of bounds after padding/trimming"}), 400
    
    try:
        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]
        
        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)
        
        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]
        
        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)
        
        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)
        
        # Step 5: Run model with aggregation details
        with torch.no_grad():
            output, aggregation_details = model(
                data_batch.x,
                data_batch.edge_index,
                data_batch.batch,
                return_aggregation_details=True,
                target_node_idx=model_target_idx
            )
        
        # Step 6: Process aggregation details
        # Map model indices back to original indices
        processed_aggregation = []
        for layer_data in aggregation_details:
            processed_layer = {
                "layer": layer_data["layer"],
                "messages": []
            }
            
            for msg in layer_data["messages"]:
                # Map model index to original index
                model_from_idx = msg["from"]
                orig_from_idx = model_from_idx - left_padding + left_trimming
                
                # Only include messages within original sequence bounds
                if 0 <= orig_from_idx < len(original_sequence):
                    processed_layer["messages"].append({
                        "from": orig_from_idx,
                        "strength": msg["strength"]
                    })
            
            processed_aggregation.append(processed_layer)
        
        # Step 7: Build graph structure for visualization
        edge_index_np = edge_index.cpu().numpy()
        edges = []
        
        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)
        
        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            
            # Only process edges within valid range
            if not (valid_start <= source < valid_end and valid_start <= target < valid_end):
                continue
            
            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming
            
            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue
            
            # Only keep one direction (source < target) to avoid duplicates
            if orig_source >= orig_target:
                continue
            
            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })
        
        # Create nodes
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {
                    "index": i,
                    "type": nuc,
                    "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"
                }
            })
        
        logger.info(f"GCN Aggregation: 节点数={len(nodes)}, 边数={len(edges)}, 层数={len(processed_aggregation)}")
        
        # Return response
        return jsonify({
            "targetNode": target_node_idx,
            "nodes": nodes,
            "edges": edges,
            "aggregationData": processed_aggregation
        }), 200
        
    except Exception as e:
        import traceback
        error_msg = f"GCN Aggregation error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


if __name__ == '__main__':
    # 运行服务器（与 Nginx 配置一致）
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
