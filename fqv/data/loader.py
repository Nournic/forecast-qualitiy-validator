"""MarketDataLoader – загрузка дневных котировок (Yahoo Finance) и индекса страха и жадности с кешированием в CSV.
Для автономных проверок есть генератор синтетического ряда с кластеризацией волатильности."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["close", "high", "low", "volume"]


class MarketDataLoader:
    def __init__(self, data_dir: str = "data/raw", start: str = "2018-01-01", end: str = "2026-11-01",
                 synthetic: bool = False):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start, self.end, self.synthetic = start, end, synthetic

    def _file(self, asset: str) -> Path:
        safe = asset.replace("^", "").replace("=", "_").replace("/", "_")
        return self.dir / (("synthetic_" if self.synthetic else "") + f"{safe}.csv")

    def load(self, asset: str) -> pd.DataFrame:
        """Возвращает таблицу close/high/low/volume с индексом-датой. Повторный вызов читает кеш."""
        f = self._file(asset)
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        else:
            df = self._synthetic(asset) if self.synthetic else self._download(asset)
            df.to_csv(f)
        df = df[COLUMNS].astype(float)
        df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)].dropna()
        if len(df) < 500:
            raise ValueError(f"{asset}: слишком короткий ряд ({len(df)} наблюдений)")
        return df

    def _download(self, asset: str) -> pd.DataFrame:
        import yfinance as yf
        raw = yf.download(asset, start=self.start, end=self.end, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"Yahoo Finance не вернул данные для {asset}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[COLUMNS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        return df

    def _synthetic(self, asset: str, n: int = 2600) -> pd.DataFrame:
        """GARCH(1,1) с редкими сменами режима; объём растёт вместе с амплитудой движения."""
        import zlib
        rng = np.random.default_rng(zlib.crc32(asset.encode()))
        omega, alpha, beta = 2e-5, 0.10, 0.85
        var, r = np.empty(n), np.empty(n)
        var[0] = omega / (1 - alpha - beta)
        regime = 1.0
        eps = 0.0
        for t in range(n):
            if rng.random() < 0.004:
                regime = rng.choice([0.6, 1.0, 1.8])
            if t:
                var[t] = omega + alpha * eps ** 2 + beta * var[t - 1]
            eps = np.sqrt(var[t]) * rng.standard_t(5) / np.sqrt(5 / 3)
            r[t] = float(np.clip(eps * regime, -0.3, 0.3))
        close = 100 * np.exp(np.cumsum(r))
        rng_hl = np.abs(r) + np.sqrt(var) * np.abs(rng.normal(1.0, 0.3, n))
        high, low = close * np.exp(rng_hl / 2), close * np.exp(-rng_hl / 2)
        volume = 1e6 * np.exp(0.4 * rng.normal(size=n) + 25 * np.abs(r))
        idx = pd.bdate_range("2016-01-04", periods=n, name="date")
        return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)

    def load_fear_greed(self) -> pd.Series | None:
        """Crypto Fear & Greed Index (alternative.me), значения 0–100. При недоступности источника – None."""
        f = self.dir / "fear_greed.csv"
        if f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True)["fng"]
            return s.sort_index()
        if self.synthetic:
            return None
        try:
            import json, urllib.request
            req = urllib.request.Request("https://api.alternative.me/fng/?limit=0&format=json",
                                         headers={"User-Agent": "fqv"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))["data"]
            s = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s").normalize(): float(d["value"]) for d in data},
                          name="fng").sort_index()
            s = s[~s.index.duplicated()]
            s.index.name = "date"
            s.to_frame().to_csv(f)
            return s
        except Exception as e:  # noqa: BLE001 - сеть недоступна: работаем без сентимента
            print(f"  [предупреждение] индекс страха и жадности недоступен: {e}")
            return None
