"""LatencyAnalyzer – задержка обнаружения синтетического шока с известным моментом начала."""
from __future__ import annotations

import numpy as np


class LatencyAnalyzer:
    @staticmethod
    def delay(alarm_days: np.ndarray, alarms: np.ndarray, start: int, horizon: int) -> int | None:
        """alarm_days[i] – день, на который выдано решение alarms[i] (образец k даёт решение на день k+1).
        Шок начинается в день start. Решение на день start принято по данным до шока, поэтому отсчёт ведётся
        со дня start + 1. Возвращает число дней от начала шока до первой тревоги или None, если тревоги
        не было в течение horizon дней."""
        for d in range(start + 1, start + horizon + 1):
            hit = alarms[alarm_days == d]
            if len(hit) and hit[0]:
                return d - start
        return None

    @staticmethod
    def summarize(delays: list, horizon: int) -> dict:
        found = [d for d in delays if d is not None]
        n = len(delays)
        return {"n": n, "detected_share": len(found) / n if n else float("nan"),
                "within_1": sum(d <= 1 for d in found) / n if n else float("nan"),
                "within_3": sum(d <= 3 for d in found) / n if n else float("nan"),
                "mean_delay_detected": float(np.mean(found)) if found else float("nan"),
                "mean_delay_censored": float(np.mean([d if d is not None else horizon + 1 for d in delays])) if n
                else float("nan")}
