"""
Backward-compatible shim for main_model_onnx.

All code has been moved to rgcnformer_backend.models.onnx_compatible.
This file re-exports the public API for backward compatibility.
"""
from rgcnformer_backend.models.onnx_compatible import (
    ParallelCNNBlock,
    GCNBlock,
    HierarchicalClassQueryHeadPooling,
    RNA_ClassQuery_Model,
)

__all__ = [
    'ParallelCNNBlock',
    'GCNBlock',
    'HierarchicalClassQueryHeadPooling',
    'RNA_ClassQuery_Model',
]
