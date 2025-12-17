from .core import matching, _create_labeled_mask
from .softpq import SoftPQ
from .average_precision import AveragePrecision

def evaluate_segmentation(y_true, y_pred, thresh=0.5, iou_high=0.5, iou_low=0.05, 
              soft_pq_method='sqrt', prioritize_underseg=False):
    """
    Evaluate the segmentation performance of a given ground truth and predicted label images.
    """
    # Convert to labeled masks if needed
    y_true_labeled = _create_labeled_mask(y_true)
    y_pred_labeled = _create_labeled_mask(y_pred)
    
    # Standard metrics
    matching_results = matching(y_true_labeled, y_pred_labeled, thresh=thresh)
    results = matching_results

    # Soft Panoptic Quality
    soft_pq = SoftPQ(iou_high=iou_high, iou_low=iou_low, 
                method=soft_pq_method, prioritize_underseg=prioritize_underseg)
    results.update({'soft_pq': soft_pq.evaluate(y_true, y_pred)})

    # Mean Average Precision
    ap = AveragePrecision(y_true_labeled, y_pred_labeled, threshold=[thresh])
    ap, _, _, _ = ap.compute()
    results.update({'mean_average_precision': ap.mean()})
    
    return results