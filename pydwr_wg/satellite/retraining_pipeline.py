"""
Downstream Satellite Precipitation Model Retraining and Evaluation Pipeline.
Evaluates shallow warm-rain recall improvements when switching from GPM DPR to PTA-InNet reference labels.
"""

import numpy as np
from typing import Dict, Any, List
from pydwr_wg.constants import PRECIPITATION_CLASSES

class SatelliteRetrainingEvaluator:
    """
    Evaluates downstream geostationary satellite precipitation classification improvements
    over the Western Ghats domain.
    """

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.class_names = PRECIPITATION_CLASSES

    def compute_confusion_matrix(
        self,
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the [n_classes, n_classes] confusion matrix.
        Rows: True classes, Columns: Predicted classes.
        """
        conf_mat = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        valid = (true_labels >= 0) & (true_labels < self.n_classes) & (pred_labels >= 0) & (pred_labels < self.n_classes)

        t = true_labels[valid].ravel()
        p = pred_labels[valid].ravel()

        for true_c in range(self.n_classes):
            mask_c = (t == true_c)
            if np.any(mask_c):
                preds_for_c = p[mask_c]
                for pred_c in range(self.n_classes):
                    conf_mat[true_c, pred_c] = np.sum(preds_for_c == pred_c)

        return conf_mat

    def compute_per_class_metrics(self, conf_mat: np.ndarray) -> Dict[str, Any]:
        """
        Calculates precision, recall, and F1-score for each precipitation class.
        Highlights shallow warm-rain recall (classes 3 and 4).
        """
        metrics = {}
        total_samples = np.sum(conf_mat)

        for c in range(self.n_classes):
            name = self.class_names.get(c, f"Class_{c}")
            tp = float(conf_mat[c, c])
            fp = float(np.sum(conf_mat[:, c]) - tp)
            fn = float(np.sum(conf_mat[c, :]) - tp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            metrics[name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(np.sum(conf_mat[c, :])),
            }

        # Overall shallow warm-rain recall across classes 3 & 4
        shallow_tp = float(conf_mat[3, 3] + conf_mat[4, 4])
        shallow_fn = float(np.sum(conf_mat[3, :]) - conf_mat[3, 3] + np.sum(conf_mat[4, :]) - conf_mat[4, 4])
        shallow_recall = shallow_tp / (shallow_tp + shallow_fn) if (shallow_tp + shallow_fn) > 0 else 0.0

        metrics["shallow_warm_rain_overall_recall"] = shallow_recall
        metrics["target_spec_met"] = shallow_recall >= 0.62

        return metrics
