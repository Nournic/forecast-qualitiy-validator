"""MetricsCalculator – метрики бинарной классификации и выбор порога решения по валидации."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, brier_score_loss


class MetricsCalculator:
    @staticmethod
    def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
        """Порог, при котором F-мера на валидации максимальна. Решение «аномалия», если оценка >= порога."""
        if len(np.unique(s)) == 1 or y.sum() == 0:
            return float(np.max(s)) if len(s) else 0.0
        order = np.argsort(-s)
        ys, ss = y[order], s[order]
        tp = np.cumsum(ys)
        k = np.arange(1, len(ys) + 1)
        f1 = 2 * tp / (k + y.sum())
        last = np.r_[ss[1:] != ss[:-1], True]        # порог можно ставить только между разными значениями оценки
        f1 = np.where(last, f1, -1.0)
        return float(ss[int(np.argmax(f1))])

    @staticmethod
    def compute(y: np.ndarray, s: np.ndarray, threshold: float, probabilistic: bool = False) -> dict:
        y = np.asarray(y).astype(int)
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        both = 0 < y.sum() < len(y)
        const = len(np.unique(s)) == 1
        out = {"n": len(y), "prevalence": float(y.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
               "accuracy": (tp + tn) / len(y), "alarm_rate": float(pred.mean()),
               "mcc": float(matthews_corrcoef(y, pred)) if both and 0 < pred.sum() < len(pred) else 0.0,
               "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
               "roc_auc": (0.5 if const else float(roc_auc_score(y, s))) if both else float("nan")}
        if probabilistic:
            out["brier"] = float(brier_score_loss(y, np.clip(s, 0, 1)))
        return out
