"""DatasetBuilder – целевая переменная, скользящая проверка вперёд, нормировка и окна без утечки данных.

Обозначения: r[k] – доходность дня k; признаки F[k] известны к концу дня k; образец k предсказывает день k+1.
Наивный прогноз цены P^_{k+1} = P_k, его ошибка в логарифмической шкале e[k+1] = |r[k+1]|.
Метка образца: y_k = 1, если e[k+1] больше порога, и порог считается только по обучающей части фолда.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    index: int
    train: np.ndarray      # номера дней k обучающих образцов
    val: np.ndarray
    test: np.ndarray
    threshold: float       # порог ошибки наивного прогноза
    mean: np.ndarray       # параметры нормировки признаков (по обучающей части)
    std: np.ndarray

    def labels(self, err_next: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return (err_next[idx] > self.threshold).astype(np.float32)

    def scale(self, F: np.ndarray) -> np.ndarray:
        return (F - self.mean) / self.std


class DatasetBuilder:
    def __init__(self, n_folds: int = 5, initial_train: float = 0.5, val_fraction: float = 0.2,
                 label_quantile: float = 0.75, max_window: int = 60, warmup: int = 30):
        self.n_folds, self.initial_train, self.val_fraction = n_folds, initial_train, val_fraction
        self.q, self.max_window, self.warmup = label_quantile, max_window, warmup

    def sample_days(self, n_days: int) -> np.ndarray:
        """Дни k, для которых есть полное окно максимальной длины и известен следующий день."""
        first = self.warmup + self.max_window - 1
        return np.arange(first, n_days - 1)

    @staticmethod
    def error_next(r: np.ndarray) -> np.ndarray:
        """err_next[k] = |r[k+1]| – ошибка наивного прогноза на день k+1 (последний элемент не определён)."""
        e = np.full(len(r), np.nan)
        e[:-1] = np.abs(r[1:])
        return e

    def folds(self, F: np.ndarray, r: np.ndarray) -> list[Fold]:
        days = self.sample_days(len(r))
        err_next = self.error_next(r)
        n = len(days)
        start = int(n * self.initial_train)
        block = (n - start) // self.n_folds
        out = []
        for f in range(self.n_folds):
            a = start + f * block
            b = n if f == self.n_folds - 1 else a + block
            trainval, test = days[:a - 1], days[a:b]          # один образец пропуска между обучением и тестом
            n_val = int(len(trainval) * self.val_fraction)
            train, val = trainval[:-n_val], trainval[-n_val:]
            thr = float(np.quantile(err_next[train], self.q))
            last = train[-1]
            # статистики нормировки – по дням до конца обучающей части
            mean = np.nanmean(F[:last + 1], axis=0)
            std = np.nanstd(F[:last + 1], axis=0)
            std = np.where(std < 1e-12, 1.0, std)
            out.append(Fold(f, train, val, test, thr, mean, std))
        return out

    @staticmethod
    def windows(Fz: np.ndarray, idx: np.ndarray, window: int) -> np.ndarray:
        """Тензор формы (число образцов, длина окна, число признаков): окно заканчивается днём k."""
        offs = np.arange(-window + 1, 1)
        return Fz[idx[:, None] + offs[None, :]].astype(np.float32)

    @staticmethod
    def tabular(Fz: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Табличные признаки для эталонов: последний день, средние за 5 и 20 дней."""
        last = Fz[idx]
        m5 = np.stack([Fz[k - 4:k + 1].mean(axis=0) for k in idx])
        m20 = np.stack([Fz[k - 19:k + 1].mean(axis=0) for k in idx])
        return np.hstack([last, m5, m20])
