# """
# Segmentation evaluation metrics package.

# This package provides a unified interface for computing various segmentation metrics.
# """

# from .core import (
#     evaluate,  # Unified evaluation endpoint
#     matching,
#     evaluate_segmentation,
#     average_precision,
#     compute_standard_pq,
#     getIoUvsThreshold,
#     _proposed_sqrt,
#     _proposed_log,
#     _proposed_linear,
#     # Helper functions (for backward compatibility)
#     precision,
#     recall,
#     accuracy,
#     f1,
#     intersection_over_union,
#     compute_iou_matrix,
#     compute_f1_score,
# )

# from .softpq import SoftPQ

# __all__ = [
#     'evaluate',  # Main unified endpoint
#     'matching',
#     'evaluate_segmentation',
#     'average_precision',
#     'compute_standard_pq',
#     'getIoUvsThreshold',
#     'SoftPQ',
#     '_proposed_sqrt',
#     '_proposed_log',
#     '_proposed_linear',
#     # Helper functions
#     'precision',
#     'recall',
#     'accuracy',
#     'f1',
#     'intersection_over_union',
#     'compute_iou_matrix',
#     'compute_f1_score',
# ]

