"""Дополнительная статистика по готовым результатам (обучение не повторяется).

Сравнивает каждую сеть с каждой другой сетью и с каждым эталоном: критерий Уилкоксона с поправкой Холма на всё
семейство сравнений, число инструментов с тем же знаком разности и бутстреп-интервал с повторной выборкой
инструментов целиком (блоки одного инструмента зависимы). Запуск: python -m experiments.pairwise
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fqv.evaluation.stats import holm, wilcoxon_pairs

RESULTS = Path(__file__).resolve().parents[1] / "results"
NETS = ("cnn", "gru", "hybrid")


def cluster_ci(d: np.ndarray, groups: list, B: int = 10000, seed: int = 0) -> tuple:
    """Интервал для средней разности: повторная выборка инструментов, а не отдельных блоков."""
    names = sorted(set(groups))
    per = np.array([d[[g == n for g in groups]].mean() for n in names])
    rng = np.random.default_rng(seed)
    means = per[rng.integers(0, len(per), size=(B, len(per)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), per.tolist()


def main() -> None:
    s = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assets = [u[0] for u in s["units"]]
    base = [m for m in s["methods"] if m not in NETS and m != "prior"]
    pairs = [("gru", "cnn"), ("hybrid", "cnn"), ("gru", "hybrid")] + [(n, b) for n in NETS for b in base]
    out = {"n_pairs": len(pairs), "n_units": len(assets), "assets": sorted(set(assets))}
    for metric in ("pr_auc", "roc_auc"):
        t = {m: np.asarray(v, dtype=float) for m, v in s["unit_tables"][metric].items()}
        info = wilcoxon_pairs(t, pairs)                      # поправка Холма внутри – на все пары сразу
        assert info == {k: {**v, "p_holm": holm({q: w["p"] for q, w in info.items()})[k]} for k, v in info.items()}
        for a, b in pairs:
            lo, hi, per = cluster_ci(t[a] - t[b], assets)
            info[f"{a} vs {b}"].update({"cluster_ci": [lo, hi], "per_asset": per,
                                        "assets_pos": int(sum(x > 0 for x in per))})
        out[metric] = info
    (RESULTS / "pairwise.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in out["pr_auc"].items():
        print(f"{k:18s} d={v['mean_diff']:+.3f} блоков {v['wins']}/{out['n_units']} инстр. {v['assets_pos']}/5 "
              f"p={v['p']:.4f} p_holm={v['p_holm']:.4f} CI=[{v['cluster_ci'][0]:+.3f}; {v['cluster_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
