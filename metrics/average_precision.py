import numpy as np
from .core import _compute_iou_matrix, _count_true_positives

class AveragePrecision:
    def __init__(self, masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
        self.masks_true = masks_true
        self.masks_pred = masks_pred
        self.threshold = threshold

    def compute(self):
        return self._compute_average_precision()


    def _compute_iou(self, mask_true, mask_pred):
        return _compute_iou_matrix(mask_true, mask_pred)

    def _compute_tp(self, iou_full, threshold):
        return _count_true_positives(iou_full, threshold)

    def _compute_fp_fn(self, n_true, n_pred, tp):
        fp = n_pred - tp
        fn = n_true - tp
        return fp, fn

    def _evaluate_single(self, mask_true, mask_pred, thresholds):
        n_true = np.max(mask_true)
        n_pred = np.max(mask_pred)

        ap_row = np.zeros(len(thresholds), np.float32)
        tp_row = np.zeros(len(thresholds), np.float32)
        fp_row = np.zeros(len(thresholds), np.float32)
        fn_row = np.zeros(len(thresholds), np.float32)

        if n_pred > 0:
            iou_full = self._compute_iou(mask_true, mask_pred)

            for k, th in enumerate(thresholds):
                tp = self._compute_tp(iou_full, th)
                fp, fn = self._compute_fp_fn(n_true, n_pred, tp)
                tp_row[k], fp_row[k], fn_row[k] = tp, fp, fn
                ap_row[k] = tp / (tp + fp + fn)

        return ap_row, tp_row, fp_row, fn_row

    def _compute_average_precision(self):
        masks_true = self.masks_true
        masks_pred = self.masks_pred
        thresholds = self.threshold

        # Normalize inputs
        single = False
        if not isinstance(masks_true, list):
            masks_true = [masks_true]
            masks_pred = [masks_pred]
            single = True
        if not isinstance(thresholds, (list, np.ndarray)):
            thresholds = [thresholds]

        ap = []
        tp = []
        fp = []
        fn = []

        for m_true, m_pred in zip(masks_true, masks_pred):
            ap_row, tp_row, fp_row, fn_row = self._evaluate_single(m_true, m_pred, thresholds)
            ap.append(ap_row)
            tp.append(tp_row)
            fp.append(fp_row)
            fn.append(fn_row)

        ap, tp, fp, fn = map(lambda x: np.stack(x), (ap, tp, fp, fn))

        if single:
            return ap[0], tp[0], fp[0], fn[0]

        return ap, tp, fp, fn
