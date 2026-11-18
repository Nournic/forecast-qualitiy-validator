"""Замер быстродействия сетей при одинаковой длине окна.

В основном эксперименте у 1D-CNN окно 10 дней, у GRU и гибрида 30, поэтому разница в скорости частично
объясняется длиной входа. Здесь каждая сеть обучается и замеряется с окнами 10 и 30 дней (остальные
гиперпараметры – выбранные в основном эксперименте) на первом тестовом блоке каждого инструмента,
один поток процессора, первое начальное значение генератора.

Запуск: python -m experiments.bench_window
Результат: results/bench_window.json и таблица в консоли.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fqv.core.config import ExperimentConfig, NetConfig
from fqv.core.runner import prepare, fit_eval
from fqv.models.registry import ModelRegistry, TITLES

RESULTS = Path("results")
WINDOWS = (10, 30)


def main() -> None:
    torch.set_num_threads(1)
    cfg = ExperimentConfig()
    tuned = {k: NetConfig(**v) for k, v in json.loads((RESULTS / "tuned.json").read_text(encoding="utf-8")).items()}
    rows = []
    for asset in cfg.assets:
        series, fb, F, db, folds, err_next = prepare(cfg, asset, cfg.features)
        fold = folds[0]
        Fz, r = fold.scale(F), series["r"]
        for net in cfg.nets:
            for w in WINDOWS:
                nc = tuned[net].with_(window=w)
                m = ModelRegistry.create(net, len(cfg.features), nc, cfg.seeds[0])
                _, _, test, t = fit_eval(m, Fz, r, fold, err_next)
                epochs = len(m.session.history["val_loss"])
                inf = m.inference_time(Fz, fold.test)
                rows.append({"asset": asset, "net": net, "window": w, "tuned_window": tuned[net].window,
                             "n_params": m.n_params, "fit_time": t, "epochs_run": epochs,
                             "epoch_time": m.session.train_time / epochs, **inf,
                             "test_pr_auc": test["pr_auc"], "test_roc_auc": test["roc_auc"]})
                print(f"{asset:8s} {net:7s} окно {w:2d}: обучение {t:5.2f} с за {epochs} эпох, "
                      f"одно предсказание {inf['single_ms']:.3f} мс", flush=True)
    summ = {}
    for net in cfg.nets:
        for w in WINDOWS:
            rr = [x for x in rows if x["net"] == net and x["window"] == w]
            summ[f"{net}_w{w}"] = {k: float(np.mean([x[k] for x in rr])) for k in
                                   ("n_params", "fit_time", "epochs_run", "epoch_time", "single_ms",
                                    "batch_ms_per_sample", "test_pr_auc", "test_roc_auc")}
    (RESULTS / "bench_window.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False,
                                                          indent=1), encoding="utf-8")
    print(f"\n{'сеть':<10}{'окно':>5}{'парам.':>8}{'обуч., с':>10}{'эпох':>6}{'эпоха, мс':>11}"
          f"{'1 предск., мс':>15}{'пачка, мс/окно':>16}")
    for k, v in summ.items():
        net, w = k.rsplit("_w", 1)
        print(f"{TITLES[net]:<10}{w:>5}{v['n_params']:>8.0f}{v['fit_time']:>10.2f}{v['epochs_run']:>6.1f}"
              f"{1000 * v['epoch_time']:>11.1f}{v['single_ms']:>15.3f}{v['batch_ms_per_sample']:>16.4f}")
    print("\nзаписано: results/bench_window.json")


if __name__ == "__main__":
    main()
