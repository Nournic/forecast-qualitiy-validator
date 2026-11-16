"""Статистическая обработка: бутстреп-интервалы, критерии Уилкоксона и Фридмана, критическая разность Немени."""
from __future__ import annotations

import numpy as np
from scipy import stats

# критические значения q_0,05 для критерия Немени (Demsar, 2006), k = 2 … 12 методов
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268}


def bootstrap_ci(x, B: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple:
    """Перцентильный доверительный интервал для среднего (повторная выборка единиц наблюдения)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvalues: dict) -> dict:
    """Поправка Холма на множественные сравнения."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def wilcoxon_pairs(table: dict, pairs: list) -> dict:
    """Парный критерий Уилкоксона. table[метод] – значения метрики по одним и тем же единицам наблюдения."""
    raw, info = {}, {}
    for a, b in pairs:
        x, y = np.asarray(table[a]), np.asarray(table[b])
        d = x - y
        key = f"{a} vs {b}"
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
        raw[key] = p
        info[key] = {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                     "p": p, "ci": bootstrap_ci(d)}
    adj = holm(raw)
    for k in info:
        info[k]["p_holm"] = adj[k]
    return info


def friedman_ranks(table: dict) -> dict:
    """Критерий Фридмана и средние ранги (1 – лучший метод; метрика «чем больше, тем лучше»)."""
    names = list(table)
    M = np.column_stack([table[n] for n in names])
    ranks = stats.rankdata(-M, axis=1)
    stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    return {"names": names, "mean_rank": dict(zip(names, ranks.mean(axis=0).tolist())), "statistic": float(stat),
            "p": float(p), "n_units": int(M.shape[0]), "cd": nemenyi_cd(len(names), M.shape[0])}


def nemenyi_cd(k: int, n: int) -> float:
    """Критическая разность средних рангов (уровень значимости 0,05)."""
    return float(_Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n))) if k in _Q05 else float("nan")
