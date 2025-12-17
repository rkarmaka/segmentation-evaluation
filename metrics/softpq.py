import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label
from .core import _compute_iou_matrix, _compute_f1_score


class SoftPQ:
    """
    This class implements the SoftPQ metric.
    It is a generalized version of the PQ metric that introduces soft-matching.
    
    Control parameters:
        iou_high: float, default=0.5 -> Uses to identify the hard matches
        iou_low: float, default=0.05 -> Uses to identify the soft matches
        method: str, default='sqrt' -> Uses to identify the penalty function
        prioritize_underseg: bool, default=False -> Prioritize undersegmentation

    The metric is computed as follows:
        - Compute the intersection over union (IoU) matrix between the ground truth and the predicted masks
        - Use the iou_high to identify the hard matches
        - If undersegmentation is prioritized, compute softmatches for the predicted masks
        - If oversegmentation is prioritized, compute softmatches for the ground truth masks
        - Use the iou_low to identify the soft matches
        - Define the penalty function
        - Based on soft matches, compute modified mean true score
        - Compute the true positives
        - Compute modified false positives and false negatives
        - Compute the F1 score
        - Compute the soft PQ score
    """

    def __init__(self, iou_high=0.5, iou_low=0.05, method='sqrt', prioritize_underseg=False):
        """
        Initialize the SoftPQ class

        Args:
            iou_high: float, default=0.5 -> Uses to identify the hard matches
            iou_low: float, default=0.05 -> Uses to identify the soft matches
            method: str, default='sqrt' -> Uses to identify the penalty function
            prioritize_underseg: bool, default=False -> Prioritize undersegmentation
        """
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.method = method
        self.prioritize_underseg = prioritize_underseg

    def _penalty(self, count, mode='sqrt'):
        """
        Compute the penalty function

        Args:
            count: int -> Number of soft matches
            mode: str, default='sqrt' -> Uses to identify the penalty function

        Returns:
            penalty: float -> Penalty function
        """
        penalty_map = {
            'sqrt': np.sqrt(count + 1),
            'log': np.maximum(1.0, np.log(count + 1)),
            'linear': count + 1
        }
        if mode not in penalty_map:
            raise ValueError(f"Unknown penalty mode: {mode}")
        return penalty_map[mode]

    def _compute_mean_true_score(self, iou_matrix, iou_thresh_high, iou_thresh_low, prioritize_underseg=False):
        """
        Compute the mean true score

        Args:
            iou_matrix: ND-array, int -> IoU matrix
            iou_thresh_high: float -> IoU threshold for hard matches
            iou_thresh_low: float -> IoU threshold for soft matches
            prioritize_underseg: bool, default=False -> Prioritize undersegmentation

        Returns:
            soft_pq_score: float -> Mean true score
            num_soft_matches: int -> Number of soft matches
        """
        if prioritize_underseg:
            iou_matrix = iou_matrix.T  # transpose to prioritize predicted-to-GT matches

        soft_pq_score = 0.0
        num_soft_matches = 0
        iou_core = iou_matrix[1:, 1:]
        num_rows, _ = iou_core.shape

        for row_index in range(num_rows):
            iou_row = iou_core[row_index]
            soft_match_mask = (iou_row > iou_thresh_low) & (iou_row < iou_thresh_high)
            num_soft_for_row = np.count_nonzero(soft_match_mask)
            matched_mask = iou_row > iou_thresh_high

            if np.any(matched_mask):
                if num_soft_for_row > 0:
                    num_soft_matches += num_soft_for_row
                penalty = self._penalty(num_soft_for_row, self.method)
                soft_pq_score += iou_row[matched_mask][0] + np.sum(iou_row[soft_match_mask] / penalty)
            else:
                penalty = self._penalty(num_soft_for_row, self.method)
                soft_pq_score += np.sum(iou_row[soft_match_mask] / penalty)

        return soft_pq_score, num_soft_matches

    def _count_true_positives(self, iou_matrix, iou_threshold):
        """
        Count the true positives
        Uses shared helper function logic.

        Args:
            iou_matrix: ND-array, int -> IoU matrix
            iou_threshold: float -> IoU threshold
        
        Returns:
            true_positives: int -> Number of true positives
        """
        iou_core = iou_matrix[1:, 1:]
        num_gt, num_pred = iou_core.shape
        max_matches = min(num_gt, num_pred)

        if max_matches == 0:
            return 0

        cost_matrix = -(iou_core >= iou_threshold).astype(float) - iou_core / (2 * max_matches)
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        assert len(gt_indices) == len(pred_indices) == max_matches

        valid_matches = iou_core[gt_indices, pred_indices] >= iou_threshold
        return np.count_nonzero(valid_matches)

    def evaluate(self, ground_truth_mask, predicted_mask):
        """
        Evaluate the SoftPQ metric

        Args:
            ground_truth_mask: ND-array, int -> Ground truth mask
            predicted_mask: ND-array, int -> Predicted mask
        
        Returns:
            soft_pq_score: float -> SoftPQ score
        """
        gt_labels = label(ground_truth_mask)
        pred_labels = label(predicted_mask)

        iou_matrix = _compute_iou_matrix(gt_labels, pred_labels)
        num_gt_objects, num_pred_objects = iou_matrix[1:, 1:].shape

        if num_gt_objects == 0 or num_pred_objects == 0:
            return 0.0

        mean_true_score, num_soft_matches = self._compute_mean_true_score(
            iou_matrix,
            self.iou_high,
            self.iou_low,
            self.prioritize_underseg
        )

        true_positives = self._count_true_positives(iou_matrix, self.iou_high)

        if self.prioritize_underseg:
            false_positives = max(num_pred_objects - true_positives, 0)
            false_negatives = max(num_gt_objects - true_positives - num_soft_matches, 0)
        else:
            false_positives = max(num_pred_objects - true_positives - num_soft_matches, 0)
            false_negatives = max(num_gt_objects - true_positives, 0)

        f1_score = _compute_f1_score(int(true_positives), int(false_positives), int(false_negatives))

        normalization_factor = true_positives if true_positives > 0 else max(num_gt_objects, 1)
        mean_true_score /= normalization_factor

        if f1_score == 0:
            return mean_true_score
        else:
            return mean_true_score*f1_score
