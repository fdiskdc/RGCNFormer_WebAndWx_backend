"""
Backward-compatible shim for main_model.

All code has been moved to rgcnformer_backend.models.rgcnformer.
This file re-exports the public API for backward compatibility.
"""
from rgcnformer_backend.models.rgcnformer import (
    ParallelCNNBlock,
    GCNBlock,
    ClassQueryHead,
    ClassQueryHeadPooling,
    HierarchicalClassQueryHeadPooling,
    RNA_ClassQuery_Model,
)

__all__ = [
    'ParallelCNNBlock',
    'GCNBlock',
    'ClassQueryHead',
    'ClassQueryHeadPooling',
    'HierarchicalClassQueryHeadPooling',
    'RNA_ClassQuery_Model',
]
