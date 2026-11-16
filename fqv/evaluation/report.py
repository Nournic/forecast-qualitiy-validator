"""ComparisonReport – сводка результатов: средние метрики с доверительными интервалами, статистические
критерии, влияние гиперпараметров, вычислительные затраты, стресс-тест, важность признаков, абляция."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..models.registry import NET_NAMES
from .latency import LatencyAnalyzer
from .stats import bootstrap_ci, friedman_ranks, wilcoxon_pairs

METRICS = ("pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "alarm_rate")


def _mean(x):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return float(np.mean(x)) if x else float("nan")


class ComparisonReport:
    def __init__(self, cfg, tuning: list, main: list, ablation: list):
        self.cfg, self.tuning, self.main, self.ablation = cfg, tuning, main, ablation

    # единица наблюдения – пара (актив, фолд); для сетей метрика усредняется по запускам
    def unit_table(self, metric: str, part: str = "test") -> tuple:
        cells = defaultdict(list)
        for a in self.main:
            for rec in a["records"]:
                cells[(rec["method"], a["asset"], rec["fold"])].append(rec[part][metric])
        units = sorted({(k[1], k[2]) for k in cells})
        methods = list(self.cfg.nets) + list(self.cfg.baselines)
        table = {m: [_mean(cells[(m, a, f)]) for a, f in units] for m in methods}
        return table, units

    def build(self) -> dict:
        cfg = self.cfg
        out = {"methods": list(cfg.nets) + list(cfg.baselines), "data": [], "overall": {}, "per_asset": {},
               "seed_std": {}}
        for a in self.main:
            out["data"].append({"asset": a["asset"], "n_days": a["n_days"], "first_day": a["first_day"],
                                "last_day": a["last_day"], "folds": a["folds"]})
        tables = {}
        for metric in METRICS:
            table, units = self.unit_table(metric)
            tables[metric] = table
            out["overall"][metric] = {m: {"mean": _mean(v), "ci": bootstrap_ci(v)} for m, v in table.items()}
        out["units"] = [list(u) for u in units]
        out["unit_tables"] = {m: tables[m] for m in ("pr_auc", "roc_auc", "f1")}
        out["prevalence_test"] = _mean([f["prevalence_test"] for a in self.main for f in a["folds"]])
        for metric in ("pr_auc", "roc_auc", "f1"):
            out["per_asset"][metric] = {
                m: {a: _mean([v for v, (ua, _) in zip(tables[metric][m], units) if ua == a]) for a in cfg.assets}
                for m in tables[metric]}
        # разброс между запусками сетей
        for net in cfg.nets:
            stds = defaultdict(list)
            for a in self.main:
                for rec in a["records"]:
                    if rec["method"] == net:
                        stds[(a["asset"], rec["fold"])].append(rec["test"]["pr_auc"])
            out["seed_std"][net] = _mean([np.std(v) for v in stds.values()])
        # статистические критерии
        out["friedman"] = {m: friedman_ranks(tables[m]) for m in ("pr_auc", "roc_auc")}
        base_names = [b for b in cfg.baselines if b != "prior"]
        out["wilcoxon"] = {}
        for metric in ("pr_auc", "roc_auc"):
            t = tables[metric]
            best_net = max(cfg.nets, key=lambda m: _mean(t[m]))
            best_base = max(base_names, key=lambda m: _mean(t[m]))
            nets = list(cfg.nets)
            pairs = [(nets[i], nets[j]) for i in range(len(nets)) for j in range(i + 1, len(nets))]
            pairs += [(n, best_base) for n in nets] + [(n, "prior") for n in nets]
            out["wilcoxon"][metric] = {"best_net": best_net, "best_baseline": best_base,
                                       "pairs": wilcoxon_pairs(t, pairs)}
        out["tuning"] = self._tuning()
        out["efficiency"] = self._efficiency()
        out["stress"] = self._stress()
        out["importance"] = self._importance()
        out["ablation"] = self._ablation(tables)
        out["history"] = {a["asset"]: a["history"] for a in self.main}
        return out

    def _tuning(self) -> dict:
        rows = [r for t in self.tuning for r in t["rows"]]
        base = self.cfg.net
        res = {"n_trainings": len(rows), "marginal": {}, "configs": {}}
        for net in self.cfg.nets:
            mine = [r for r in rows if r["net"] == net]
            by_cfg = defaultdict(list)
            for r in mine:
                by_cfg[r["config"]].append(r["val_pr_auc"])
            res["configs"][net] = {k: _mean(v) for k, v in by_cfg.items()}
            res["marginal"][net] = {}
            for factor, values in self.cfg.grid.items():
                others = [f for f in self.cfg.grid if f != factor]
                line = {}
                for v in values:
                    sel = [r for r in mine if r[factor] == v and all(r[o] == getattr(base, o) for o in others)]
                    line[str(v)] = {"val_pr_auc": _mean([r["val_pr_auc"] for r in sel]),
                                    "val_roc_auc": _mean([r["val_roc_auc"] for r in sel]),
                                    "time": _mean([r["time"] for r in sel]),
                                    "n_params": _mean([r["n_params"] for r in sel])}
                res["marginal"][net][factor] = line
        return res

    def _efficiency(self) -> dict:
        res = {}
        for m in list(self.cfg.nets) + list(self.cfg.baselines):
            recs = [r for a in self.main for r in a["records"] if r["method"] == m]
            row = {"fit_time": _mean([r["fit_time"] for r in recs])}
            if m in NET_NAMES:
                row.update({"n_params": int(recs[0]["n_params"]), "epochs": _mean([r["epochs"] for r in recs]),
                            "epochs_run": _mean([r["epochs_run"] for r in recs]),
                            "single_ms": _mean([a["inference"][m]["single_ms"] for a in self.main]),
                            "batch_ms_per_sample": _mean([a["inference"][m]["batch_ms_per_sample"]
                                                          for a in self.main])})
            res[m] = row
        return res

    def _stress(self) -> dict:
        res = {}
        for scen in ("shock", "drift"):
            by = defaultdict(list)
            for a in self.main:
                for r in a["stress"]:
                    if r["scenario"] == scen:
                        by[r["method"]].append(r["delay"])
            res[scen] = {m: LatencyAnalyzer.summarize(d, self.cfg.stress_length) for m, d in by.items()}
        return res

    def _importance(self) -> dict:
        res = {"features": list(self.cfg.features), "permutation": {}, "saliency_by_lag": {}}
        for net in self.cfg.nets:
            drops = [i["drop_pr_auc"] for a in self.main for i in a["importance"] if i["method"] == net]
            res["permutation"][net] = np.nanmean(np.array(drops, dtype=float), axis=0).tolist()
            sal = np.array([s["by_lag"] for a in self.main for s in a["saliency"] if s["method"] == net])
            sal = sal / sal.sum(axis=1, keepdims=True)
            res["saliency_by_lag"][net] = sal.mean(axis=0).tolist()
        return res

    def _ablation(self, tables: dict) -> dict:
        cells = defaultdict(list)
        for a in self.ablation:
            for r in a["rows"]:
                cells[(r["set"], r["method"], r["asset"], r["fold"])].append(
                    (r["test"]["pr_auc"], r["test"]["roc_auc"]))
        # полный набор признаков берётся из основного эксперимента
        for a in self.main:
            for r in a["records"]:
                if r["method"] in NET_NAMES or r["method"] == "logreg":
                    cells[("full", r["method"], a["asset"], r["fold"])].append(
                        (r["test"]["pr_auc"], r["test"]["roc_auc"]))
        sets = sorted({k[0] for k in cells})
        res = {"sets": sets, "all_assets": {}, "crypto_only": {}}
        crypto = [a for a in self.cfg.assets if a.endswith("-USD")]
        for scope, assets in (("all_assets", list(self.cfg.assets)), ("crypto_only", crypto)):
            for s in sets:
                if s == "full+fng" and scope == "all_assets":
                    continue
                res[scope][s] = {}
                for m in list(self.cfg.nets) + ["logreg"]:
                    vals = [np.nanmean(np.array(v), axis=0) for k, v in cells.items()
                            if k[0] == s and k[1] == m and k[2] in assets]
                    if vals:
                        arr = np.array(vals)
                        res[scope][s][m] = {"pr_auc": float(np.nanmean(arr[:, 0])),
                                            "roc_auc": float(np.nanmean(arr[:, 1])), "n_units": len(vals)}
        return res
