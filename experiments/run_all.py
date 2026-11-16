"""Полный вычислительный эксперимент.

    python -m experiments.run_all                # полный прогон (данные Yahoo Finance)
    python -m experiments.run_all --quick        # быстрая проверка (2 актива, 2 фолда, несколько эпох)
    python -m experiments.run_all --synthetic    # без сети: синтетические ряды
    python -m experiments.run_all --jobs 5       # число параллельных процессов (по одному на актив)

Результаты: results/summary.json и файлы по активам. Прерванный прогон продолжается той же командой."""
from __future__ import annotations

import argparse
import os

from fqv.core.config import ExperimentConfig
from fqv.core.runner import ExperimentRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--force", action="store_true", help="пересчитать, даже если результаты уже есть")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    cfg = ExperimentConfig(synthetic=a.synthetic)
    if a.quick:
        cfg = cfg.quick()
    name = a.results or ("results" + ("_quick" if a.quick else "") + ("_synthetic" if a.synthetic else ""))
    cfg.results_dir = name
    os.makedirs(name, exist_ok=True)
    summary = ExperimentRunner(cfg, jobs=a.jobs, force=a.force).run()
    print("\nСредние по парам «актив × фолд» (тестовые блоки):")
    print(f"{'метод':<10}{'PR-AUC':>9}{'ROC-AUC':>9}{'F1':>8}")
    for m in summary["methods"]:
        o = summary["overall"]
        print(f"{m:<10}{o['pr_auc'][m]['mean']:>9.3f}{o['roc_auc'][m]['mean']:>9.3f}{o['f1'][m]['mean']:>8.3f}")
    print(f"доля аномалий в тесте: {summary['prevalence_test']:.3f}")


if __name__ == "__main__":
    main()
