"""Перестановочная важность признаков: насколько падает PR-AUC, если значения одного признака перемешать
между образцами (временная структура внутри окна сохраняется)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def permutation_importance(model, Fz: np.ndarray, r: np.ndarray, idx: np.ndarray, y: np.ndarray, n_features: int,
                           repeats: int = 3, seed: int = 0) -> list:
    if not 0 < y.sum() < len(y):
        return [float("nan")] * n_features
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.score(Fz, r, idx))
    w = model.cfg.window
    offs = np.arange(-w + 1, 1)
    out = []
    for j in range(n_features):
        drops = []
        for _ in range(repeats):
            # перестановка окон признака j между образцами: строим изменённую копию ряда «по образцам»
            perm = rng.permutation(len(idx))
            scores = np.empty(len(idx))
            for start in range(0, len(idx), 512):
                sl = slice(start, start + 512)
                block = Fz[idx[sl, None] + offs[None, :]].copy()            # (b, w, C)
                block[:, :, j] = Fz[idx[perm[sl], None] + offs[None, :], j]
                scores[sl] = _score_windows(model, block)
            drops.append(base - average_precision_score(y, scores))
        out.append(float(np.mean(drops)))
    return out


def _score_windows(model, block: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return torch.sigmoid(model.net(torch.from_numpy(block.astype(np.float32)))).numpy()
