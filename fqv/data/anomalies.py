"""AnomalyInjector – синтетические сценарии для стресс-теста. Сценарий меняет ряды только на отрезке
[start; start + length), момент начала известен точно, поэтому задержку обнаружения можно измерить."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyScenario:
    start: int
    length: int = 10
    factor: float = 3.0

    def multipliers(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LocalShock(AnomalyScenario):
    """Резкий рост амплитуды колебаний в factor раз на весь отрезок."""

    def multipliers(self) -> np.ndarray:
        return np.full(self.length, self.factor)


@dataclass
class StructuralDrift(AnomalyScenario):
    """Плавный рост амплитуды от 1 до factor."""

    def multipliers(self) -> np.ndarray:
        return np.linspace(1.0, self.factor, self.length)


class AnomalyInjector:
    @staticmethod
    def apply(series: dict, scenario: AnomalyScenario) -> dict:
        """Возвращает копию рядов с изменёнными доходностью и размахом; исходные ряды не меняются."""
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in series.items()}
        a, b = scenario.start, min(scenario.start + scenario.length, len(out["r"]))
        m = scenario.multipliers()[:b - a]
        out["r"][a:b] *= m
        out["hl"][a:b] *= m
        return out
