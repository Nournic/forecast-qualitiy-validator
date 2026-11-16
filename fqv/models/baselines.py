"""Эталонные методы (BenchmarkDetector). Все они используют только данные до дня k включительно."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..data.dataset import DatasetBuilder
from .base import AnomalyValidator


class PriorDetector(AnomalyValidator):
    """Постоянная оценка, равная доле аномалий в обучающей части (нижняя граница качества)."""
    name, probabilistic = "prior", True

    def fit(self, Fz, r, fold, y_train, y_val):
        self.p = float(np.mean(y_train))
        return self

    def score(self, Fz, r, idx):
        return np.full(len(idx), self.p)


class PersistenceDetector(AnomalyValidator):
    """«Завтра будет как сегодня»: оценка равна сегодняшней ошибке |r_k|."""
    name = "persist"

    def score(self, Fz, r, idx):
        return np.abs(r[idx])


class MovingAverageDetector(AnomalyValidator):
    """Скользящее среднее ошибок за последние 7 дней (k-6 … k). Целевой день k+1 в окно не входит."""
    name = "roll7"

    def __init__(self, window: int = 7):
        self.w = window

    def score(self, Fz, r, idx):
        a = np.abs(r)
        c = np.concatenate([[0.0], np.cumsum(a)])
        return (c[idx + 1] - c[idx + 1 - self.w]) / self.w


class EwmaDetector(AnomalyValidator):
    """Экспоненциально взвешенная волатильность (RiskMetrics, лямбда = 0,94)."""
    name = "ewma"

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    def score(self, Fz, r, idx):
        v = np.empty(len(r))
        v[0] = np.var(r[:30])
        for t in range(1, len(r)):
            v[t] = self.lam * v[t - 1] + (1 - self.lam) * r[t] ** 2
        return np.sqrt(v[idx])


class GarchDetector(AnomalyValidator):
    """GARCH(1,1): параметры оцениваются методом максимального правдоподобия по обучающей части,
    оценка – прогноз условного стандартного отклонения на день k+1."""
    name = "garch"

    @staticmethod
    def _variance(r, mu, omega, alpha, beta, v0):
        v = np.empty(len(r) + 1)
        v[0] = v0
        e2 = (r - mu) ** 2
        for t in range(len(r)):
            v[t + 1] = omega + alpha * e2[t] + beta * v[t]
        return v            # v[t+1] – прогноз дисперсии на день t+1 по данным до дня t

    def fit(self, Fz, r, fold, y_train, y_val):
        x = r[:fold.train[-1] + 1]
        self.mu, var0 = float(x.mean()), float(x.var())
        self.v0 = var0

        def unpack(p):
            a = 1 / (1 + np.exp(-p[0])) * 0.5              # alpha в (0; 0,5)
            b = 1 / (1 + np.exp(-p[1])) * (0.999 - a)      # alpha + beta < 1
            return var0 * (1 - a - b), a, b

        def nll(p):
            w, a, b = unpack(p)
            v = self._variance(x, self.mu, w, a, b, var0)[:-1]
            return 0.5 * np.sum(np.log(v) + (x - self.mu) ** 2 / v)

        best = min((minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6})
                    for p0 in ([-1.5, 2.0], [-2.5, 3.0])), key=lambda o: o.fun)
        self.omega, self.alpha, self.beta = unpack(best.x)
        return self

    def score(self, Fz, r, idx):
        v = self._variance(r, self.mu, self.omega, self.alpha, self.beta, self.v0)
        return np.sqrt(v[idx + 1])


class _Tabular(AnomalyValidator):
    def _x(self, Fz, idx):
        return DatasetBuilder.tabular(Fz, idx)


class LogisticDetector(_Tabular):
    """Логистическая регрессия на табличных признаках (последний день, средние за 5 и 20 дней)."""
    name, probabilistic = "logreg", True

    def fit(self, Fz, r, fold, y_train, y_val):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        self.m.fit(self._x(Fz, fold.train), y_train)
        return self

    def score(self, Fz, r, idx):
        return self.m.predict_proba(self._x(Fz, idx))[:, 1]


class BoostingDetector(_Tabular):
    """Градиентный бустинг над деревьями на тех же табличных признаках."""
    name, probabilistic = "gboost", True

    def fit(self, Fz, r, fold, y_train, y_val):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                class_weight="balanced", random_state=0)
        self.m.fit(self._x(Fz, fold.train), y_train)
        return self

    def score(self, Fz, r, idx):
        return self.m.predict_proba(self._x(Fz, idx))[:, 1]


class IsolationForestDetector(_Tabular):
    """Изолирующий лес: обучается без меток, оценка – степень «необычности» текущего состояния рынка."""
    name = "iforest"

    def fit(self, Fz, r, fold, y_train, y_val):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(self._x(Fz, fold.train))
        return self

    def score(self, Fz, r, idx):
        return -self.m.score_samples(self._x(Fz, idx))
