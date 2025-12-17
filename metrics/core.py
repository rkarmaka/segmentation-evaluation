import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label


#########################################################################################
# Shared Helper Functions
#########################################################################################

def _safe_divide(x, y, eps=1e-10):
    '''
    This function computes a safe divide which returns 0 if y is zero.
    '''
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out


def _label_overlap(x, y):
    '''
    This function computes the overlap between two label images.
    '''
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap


def _compute_iou_matrix(ground_truth_labels, prediction_labels):
    """
    Compute the IoU matrix between ground truth and predicted labels.
    This is a unified function that consolidates intersection_over_union, 
    _intersection_over_union, and compute_iou_matrix.
    """
    overlap = _label_overlap(ground_truth_labels, prediction_labels)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    union = n_pixels_pred + n_pixels_true - overlap
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = np.true_divide(overlap, union)
        iou_matrix[~np.isfinite(iou_matrix)] = 0.0
    
    return iou_matrix


def _create_labeled_mask(mask):
    '''
    This function creates a labeled mask from a binary mask.
    '''
    if mask.dtype == 'bool':
        mask = mask.astype('uint8')
    return label(mask)


def _compute_f1_score(tp, fp, fn):
    """
    Compute F1 score given TP, FP, FN.
    This is a unified function that consolidates f1, _f1, and compute_f1_score.
    """
    if tp > 0:
        return (2*tp)/(2*tp+fp+fn)
    return 0.0


def _compute_precision(tp, fp, fn):
    """Compute precision score."""
    return tp/(tp+fp) if tp > 0 else 0


def _compute_recall(tp, fp, fn):
    """Compute recall score."""
    return tp/(tp+fn) if tp > 0 else 0


def _compute_accuracy(tp, fp, fn):
    """Compute accuracy score."""
    return tp/(tp+fp+fn) if tp > 0 else 0


def _count_true_positives(iou_matrix, iou_threshold):
    """
    Count the true positives using optimal matching.
    This consolidates the matching logic used in multiple places.
    """
    iou_core = iou_matrix[1:, 1:]
    n_true, n_pred = iou_core.shape
    n_matched = min(n_true, n_pred)
    
    if n_matched == 0:
        return 0
    
    costs = -(iou_core >= iou_threshold).astype(float) - iou_core / (2*n_matched)
    true_ind, pred_ind = linear_sum_assignment(costs)
    
    assert n_matched == len(true_ind) == len(pred_ind)
    match_ok = iou_core[true_ind, pred_ind] >= iou_threshold
    return np.count_nonzero(match_ok)


#########################################################################################
# Standard Metrics Functions
#########################################################################################

def matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false negatives (fn)
    whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)
    thresh: float
        threshold for matching criterion (default 0.5)
    criterion: string
        matching criterion (default IoU)
    report_matches: bool
        if True, additionally calculate matched_pairs and matched_scores (note, that this returns even gt-pred pairs whose scores are below  'thresh')

    Returns
    -------
    Matching object with different metrics as attributes

    Examples
    --------
    >>> y_true = np.zeros((100,100), np.uint16)
    >>> y_true[10:20,10:20] = 1
    >>> y_pred = np.roll(y_true,5,axis = 0)

    >>> stats = matching(y_true, y_pred)
    >>> print(stats)
    Matching(criterion='iou', thresh=0.5, fp=1, tp=0, fn=1, precision=0, recall=0, accuracy=0, f1=0, n_true=1, n_pred=1, mean_true_score=0.0, mean_matched_score=0.0, panoptic_quality=0.0)

    """
    iou_matrix = _compute_iou_matrix(y_true, y_pred)
    
    # ignoring background
    scores = iou_matrix[1:,1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    def _single(thr):
        not_trivial = n_matched > 0
        if not_trivial:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= thr).astype(float) - scores / (2*n_matched)
            true_ind, pred_ind = linear_sum_assignment(costs)
            assert n_matched == len(true_ind) == len(pred_ind)
            match_ok = scores[true_ind,pred_ind] >= thr
            tp = np.count_nonzero(match_ok)
        else:
            tp = 0
            true_ind, pred_ind = [], []
            match_ok = np.array([])
        
        fp = n_pred - tp
        fn = n_true - tp

        # the score sum over all matched objects (tp)
        sum_matched_score = np.sum(scores[true_ind,pred_ind][match_ok]) if not_trivial and len(match_ok) > 0 else 0.0

        # the score average over all matched objects (tp)
        mean_matched_score = _safe_divide(sum_matched_score, tp)
        # the score average over all gt/true objects
        mean_true_score = _safe_divide(sum_matched_score, n_true)
        panoptic_quality = _safe_divide(sum_matched_score, tp+fp/2+fn/2)

        stats_dict = dict(
            criterion          = criterion,
            thresh             = thr,
            fp                 = fp,
            tp                 = tp,
            fn                 = fn,
            precision          = np.round(_compute_precision(tp,fp,fn),4).item(),
            recall             = np.round(_compute_recall(tp,fp,fn),4).item(),
            accuracy           = np.round(_compute_accuracy(tp,fp,fn),4).item(),
            f1                 = np.round(_compute_f1_score(tp,fp,fn),4).item(),
            n_true             = n_true,
            n_pred             = n_pred,
            mean_true_score    = np.round(mean_true_score,4).item(),
            mean_matched_score = np.round(mean_matched_score,4).item(),
            panoptic_quality   = np.round(panoptic_quality,4).item(),
        )
        if bool(report_matches):
            if not_trivial:
                stats_dict.update(
                    # int() to be json serializable
                    matched_pairs  = tuple((int(1+i),int(1+j)) for i,j in zip(true_ind, pred_ind)),
                    matched_scores = tuple(scores[true_ind,pred_ind]),
                    matched_tps    = tuple(map(int,np.flatnonzero(match_ok))),
                )
            else:
                stats_dict.update(
                    matched_pairs  = (),
                    matched_scores = (),
                    matched_tps    = (),
                )
        return stats_dict

    return _single(thresh) if np.isscalar(thresh) else tuple(map(_single,thresh))


def evaluate_segmentation(y_true, y_pred, thresh=0.5):
    '''
    This function evaluates the segmentation performance of a given ground truth and predicted label images.
    '''
    y_true = _create_labeled_mask(y_true)
    y_pred = _create_labeled_mask(y_pred)
    score = matching(y_true, y_pred, thresh=thresh)
    return score


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ 
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)): 
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)): 
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]): 
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]): 
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]): 
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]): 
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)")
    
    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        if n_pred[n] > 0:
            iou = _compute_iou_matrix(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _count_true_positives(_compute_iou_matrix(masks_true[n], masks_pred[n]), th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

