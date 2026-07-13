"""
中文：统一路径解析模块。所有资源路径（JSON、模型权重、LinearFold）均相对于项目根目录解析，不受启动时工作目录影响。
English: Unified path resolution. All resource paths (JSON, model weights, LinearFold) are resolved relative to the project root, regardless of the working directory at startup.
"""
import os

# Project root: the directory containing this package (mrmodn_backend/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Resource paths
JSON_DIR = os.path.join(PROJECT_ROOT, 'json')
HUMAN_JSON_PATH = os.path.join(JSON_DIR, 'human.json')
MODEL_GRAPH_JSON_PATH = os.path.join(JSON_DIR, 'model_graph.json')
MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'epoch_040.pt')
LINEARFOLD_PATH = os.path.join(PROJECT_ROOT, 'LinearFold', 'linearfold')


def get_project_root() -> str:
    """中文：获取项目根目录路径。 / English: Get the project root directory path."""
    return PROJECT_ROOT


def get_json_dir() -> str:
    """中文：获取 JSON 配置文件目录路径。 / English: Get the JSON configuration directory path."""
    return JSON_DIR


def get_human_json_path() -> str:
    """中文：获取 human.json 文件路径。 / English: Get the human.json file path."""
    return HUMAN_JSON_PATH


def get_model_graph_json_path() -> str:
    """中文：获取 model_graph.json 文件路径。 / English: Get the model_graph.json file path."""
    return MODEL_GRAPH_JSON_PATH


def get_model_checkpoint_path() -> str:
    """中文：获取模型检查点文件路径。 / English: Get the model checkpoint file path."""
    return MODEL_CHECKPOINT_PATH


def get_linearfold_path() -> str:
    """中文：获取 LinearFold 可执行文件路径。 / English: Get the LinearFold executable path."""
    return LINEARFOLD_PATH
