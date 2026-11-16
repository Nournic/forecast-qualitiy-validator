"""Общий интерфейс валидаторов: обучение на фолде и оценка «тревожности» для заданных дней."""
from __future__ import annotations

import numpy as np


class AnomalyValidator:
    name = "base"
    probabilistic = False     # выдаёт ли метод вероятность (иначе – произвольную монотонную оценку)

    def fit(self, Fz: np.ndarray, r: np.ndarray, fold, y_train: np.ndarray, y_val: np.ndarray) -> "AnomalyValidator":
        """Fz – нормированные признаки по дням, r – доходности, fold – разбиение (train/val/test – номера дней)."""
        return self

    def score(self, Fz: np.ndarray, r: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Оценка для образцов idx: чем больше, тем вероятнее, что прогноз на следующий день окажется плохим."""
        raise NotImplementedError
