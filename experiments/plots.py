"""Графики отчёта по results/summary.json.   python -m experiments.plots [каталог результатов]  ->  docs/fig_*.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fqv.models.registry import TITLES, NET_NAMES

OUT = Path("docs")
C_NET, C_BASE = "#2f6fb0", "#9aa3ad"
NET_COLORS = {"cnn": "#2f6fb0", "gru": "#d9822b", "hybrid": "#3a9a5c"}
FEAT = {"ret": "Доходность", "vol7": "Волатильность, 7 дн.", "vol30": "Волатильность, 30 дн.",
        "dvol": "Относительный объём", "range": "Внутридневной размах"}
SHORT = {"ret": "Доходность", "vol7": "Волатильность\n7 дней", "vol30": "Волатильность\n30 дней", "dvol": "Объём", "range": "Размах"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_quality(s):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    order = sorted(s["methods"], key=lambda m: s["overall"]["roc_auc"][m]["mean"])
    for ax, metric, title in zip(axes, ("pr_auc", "roc_auc"), ("PR-AUC", "ROC-AUC")):
        o = s["overall"][metric]
        mean = np.array([o[m]["mean"] for m in order])
        lo = mean - np.array([o[m]["ci"][0] for m in order])
        hi = np.array([o[m]["ci"][1] for m in order]) - mean
        ax.barh(range(len(order)), mean, xerr=[lo, hi], color=[C_NET if m in NET_NAMES else C_BASE for m in order],
                height=0.62, error_kw={"lw": 1, "capsize": 2.5})
        ref = s["prevalence_test"] if metric == "pr_auc" else 0.5
        ax.axvline(ref, color="k", lw=0.8, ls="--")
        ax.set_xlim(ref - 0.05 if metric == "roc_auc" else 0.1, None)
        ax.set_xlabel(title)
        ax.set_yticks(range(len(order)), [TITLES[m] for m in order])
        ax.grid(axis="y", visible=False)
    save(fig, "fig_quality.png")


def fig_ranks(s):
    f = s["friedman"]["pr_auc"]
    names = sorted(f["mean_rank"], key=f["mean_rank"].get)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    r = [f["mean_rank"][m] for m in names]
    ax.barh(range(len(names)), r, color=[C_NET if m in NET_NAMES else C_BASE for m in names], height=0.62)
    ax.axvline(r[0] + f["cd"], color="k", lw=0.8, ls="--")
    ax.text(r[0] + f["cd"] + 0.1, len(names) - 1, "лучший ранг +\nкритическая\nразность", va="top", fontsize=9)
    ax.set_yticks(range(len(names)), [TITLES[m] for m in names])
    ax.invert_yaxis()
    ax.set_xlabel("Средний ранг по PR-AUC (меньше – лучше)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_ranks.png")


def fig_tuning(s):
    labels = {"window": "Длина окна, дней", "hidden": "Размер скрытого слоя", "dropout": "Dropout",
              "lr": "Скорость обучения", "kernel": "Размер ядра свёртки"}
    marg = s["tuning"]["marginal"]
    factors = [f for f in labels if f in marg["cnn"]]
    fig, axes = plt.subplots(1, len(factors), figsize=(3.1 * len(factors), 3.4), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), factors):
        for net in NET_NAMES:
            if f == "kernel" and net == "gru":
                continue
            line = marg[net][f]
            ax.plot(range(len(line)), [v["val_roc_auc"] for v in line.values()], "o-", color=NET_COLORS[net],
                    label=TITLES[net], lw=1.6, ms=4)
        ax.set_xticks(range(len(line)), [str(k).replace(".", ",") for k in line])
        ax.set_xlabel(labels[f])
    np.atleast_1d(axes)[0].set_ylabel("ROC-AUC на валидации")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    save(fig, "fig_tuning.png")


def fig_importance(s):
    imp = s["importance"]
    feats = imp["features"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.26
    for i, net in enumerate(NET_NAMES):
        axes[0].bar(np.arange(len(feats)) + (i - 1) * w, imp["permutation"][net], w, color=NET_COLORS[net],
                    label=TITLES[net])
    axes[0].set_xticks(range(len(feats)), [SHORT[f] for f in feats], fontsize=9)
    axes[0].set_ylabel("Падение PR-AUC при перестановке")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="x", visible=False)
    for net in NET_NAMES:
        v = imp["saliency_by_lag"][net]
        axes[1].plot(np.arange(-len(v) + 1, 1), v, "o-", color=NET_COLORS[net], label=TITLES[net], lw=1.5, ms=3)
    axes[1].set_xlabel("День окна относительно текущего")
    axes[1].set_ylabel("Доля вклада (|градиент × вход|)")
    save(fig, "fig_importance.png")


def fig_stress(s):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    alarm = s["overall"]["alarm_rate"]
    order = sorted(s["stress"]["shock"], key=lambda m: s["stress"]["shock"][m]["mean_delay_censored"])
    for ax, scen, title in zip(axes, ("shock", "drift"), ("Резкий шок", "Плавный дрейф")):
        d = [s["stress"][scen][m]["mean_delay_censored"] for m in order]
        ax.barh(range(len(order)), d, color=[C_NET if m in NET_NAMES else C_BASE for m in order], height=0.62)
        for i, m in enumerate(order):
            ax.text(d[i] + 0.06, i, f"{alarm[m]['mean']:.2f}".replace(".", ","), va="center", fontsize=8.5)
        ax.set_yticks(range(len(order)), [TITLES[m] for m in order])
        ax.set_xlabel(f"{title}: средняя задержка, дней")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    save(fig, "fig_stress.png")


def fig_ablation(s):
    ab = s["ablation"]["all_assets"]
    sets = [("R", "Только доходность"), ("RV", "Доходность и\nволатильность"), ("full", "Полный набор")]
    methods = list(NET_NAMES) + ["logreg"]
    colors = dict(NET_COLORS, logreg=C_BASE)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = 0.2
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(sets)) + (i - 1.5) * w, [ab[k][m]["roc_auc"] for k, _ in sets], w, color=colors[m],
               label=TITLES[m])
    ax.set_xticks(range(len(sets)), [t for _, t in sets])
    ax.set_ylim(0.54, None)
    ax.set_ylabel("ROC-AUC на тестовых блоках")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_ablation.png")


def fig_history(s, asset="BTC-USD"):
    h = s["history"].get(asset) or next(iter(s["history"].values()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, net in zip(axes, NET_NAMES):
        e = range(1, len(h[net]["train_loss"]) + 1)
        ax.plot(e, h[net]["train_loss"], color=NET_COLORS[net], lw=1.6, label="обучение")
        ax.plot(e, h[net]["val_loss"], color=NET_COLORS[net], lw=1.6, ls="--", label="валидация")
        ax.axvline(int(np.argmin(h[net]["val_loss"])) + 1, color="k", lw=0.7, ls=":")
        ax.set_title(TITLES[net], fontsize=11)
        ax.set_xlabel("Эпоха")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Функция потерь")
    save(fig, "fig_history.png")


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    s = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    for f in (fig_quality, fig_ranks, fig_tuning, fig_importance, fig_stress, fig_ablation, fig_history):
        f(s)
    print("рисунки сохранены в", OUT.resolve())


if __name__ == "__main__":
    main()
