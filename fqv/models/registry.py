"""ModelRegistry – создание валидатора по имени."""
from __future__ import annotations

from .baselines import (PriorDetector, PersistenceDetector, MovingAverageDetector, EwmaDetector, GarchDetector,
                        LogisticDetector, BoostingDetector, IsolationForestDetector)

NET_NAMES = ("cnn", "gru", "hybrid")
TITLES = {"cnn": "1D-CNN", "gru": "GRU", "hybrid": "CNN+GRU", "prior": "Априорная доля", "persist": "Вчерашняя ошибка",
          "roll7": "Скользящее среднее", "ewma": "EWMA", "garch": "GARCH(1,1)", "logreg": "Логистическая регрессия",
          "gboost": "Градиентный бустинг", "iforest": "Изолирующий лес"}


class ModelRegistry:
    _baselines = {c.name: c for c in (PriorDetector, PersistenceDetector, MovingAverageDetector, EwmaDetector,
                                      GarchDetector, LogisticDetector, BoostingDetector, IsolationForestDetector)}

    @classmethod
    def names(cls) -> tuple:
        return NET_NAMES + tuple(cls._baselines)

    @classmethod
    def create(cls, name: str, n_features: int = 0, cfg=None, seed: int = 0):
        if name in NET_NAMES:
            from .neural import NeuralValidator      # torch импортируется только при необходимости
            return NeuralValidator(name, cfg, n_features, seed)
        if name not in cls._baselines:
            raise KeyError(f"неизвестная модель: {name}")
        return cls._baselines[name]()
