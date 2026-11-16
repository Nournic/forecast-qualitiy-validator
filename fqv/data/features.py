"""FeatureBuilder – причинные признаки: значение в день t зависит только от данных до дня t включительно."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("ret", "vol7", "vol30", "dvol", "range", "fng")


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=w).std(ddof=0).to_numpy()


class FeatureBuilder:
    """Строит матрицу признаков по рядам доходности, объёма и внутридневного размаха.

    ret   – логарифмическая доходность r_t;
    vol7  – стандартное отклонение доходности за 7 дней;
    vol30 – то же за 30 дней;
    dvol  – логарифм отношения объёма к среднему объёму за 30 дней;
    range – логарифм отношения максимальной цены дня к минимальной;
    fng   – индекс страха и жадности, приведённый к [0; 1] (только для криптовалют).
    """

    def __init__(self, names=("ret", "vol7", "vol30", "dvol", "range")):
        unknown = set(names) - set(FEATURES)
        if unknown:
            raise ValueError(f"неизвестные признаки: {unknown}")
        self.names = tuple(names)

    @staticmethod
    def base_series(df: pd.DataFrame, fng: pd.Series | None = None) -> dict:
        """Исходные ряды, выровненные по дням доходности (первый день котировок отбрасывается)."""
        close = df["close"].to_numpy()
        out = {"dates": df.index[1:], "r": np.diff(np.log(close)),
               "volume": df["volume"].to_numpy()[1:], "hl": np.log(df["high"].to_numpy()[1:] / df["low"].to_numpy()[1:])}
        if fng is not None:
            # значение индекса за день t публикуется утром дня t, поэтому оно известно к концу дня t
            out["fng"] = fng.reindex(df.index[1:]).ffill().to_numpy() / 100.0
        return out

    def build(self, s: dict) -> np.ndarray:
        """Матрица признаков формы (число дней, число признаков). Первые дни содержат NaN (разогрев окон)."""
        r = s["r"]
        cols = []
        for name in self.names:
            if name == "ret":
                cols.append(r)
            elif name == "vol7":
                cols.append(_rolling_std(r, 7))
            elif name == "vol30":
                cols.append(_rolling_std(r, 30))
            elif name == "dvol":
                v = np.maximum(s["volume"], 1.0)
                mean30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
                cols.append(np.log(v / mean30))
            elif name == "range":
                cols.append(s["hl"])
            elif name == "fng":
                if "fng" not in s:
                    raise ValueError("ряд fng не загружен")
                cols.append(s["fng"])
        return np.column_stack(cols).astype(np.float64)
