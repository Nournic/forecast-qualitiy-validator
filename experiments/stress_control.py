"""Контроль к стресс-тесту: те же дни начала вставок, но исходный ряд без шока.

Для каждого инструмента и тестового блока модели обучаются заново так же, как в основном эксперименте
(эталоны и сети с первым начальным значением генератора, те же конфигурации и пороги тревоги). Дни начала
вставок берутся из results/main_<инструмент>.json. Для каждой вставки считаются:
  - задержка первой тревоги на изменённом ряду (проверка, что прогон воспроизводит основной эксперимент);
  - задержка первой тревоги на исходном ряду с того же дня (контроль);
  - доля дней с тревогой в окне вставки (10 дней) на изменённом и на исходном ряду.

Запуск: python -m experiments.stress_control [--jobs 5] [--no-nets]
Результат: results/stress_control.json и таблица в консоли.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fqv.core.config import ExperimentConfig, NetConfig
from fqv.core.runner import prepare, fit_eval, _safe
from fqv.data import AnomalyInjector, LocalShock, StructuralDrift
from fqv.evaluation.latency import LatencyAnalyzer
from fqv.models.registry import ModelRegistry, TITLES

RESULTS = Path("results")
SCENARIOS = {"shock": LocalShock, "drift": StructuralDrift}


def _alarm_share(days, alarms, start, L):
    sel = (days >= start + 1) & (days <= start + L)
    return float(alarms[sel].mean())


def control_asset(args) -> dict:
    asset, use_nets = args
    cfg = ExperimentConfig()
    if use_nets:
        import torch
        torch.set_num_threads(1)
    main = json.loads((RESULTS / f"main_{_safe(asset)}.json").read_text(encoding="utf-8"))
    tuned = {k: NetConfig(**v) for k, v in json.loads((RESULTS / "tuned.json").read_text(encoding="utf-8")).items()}
    series, fb, F, db, folds, err_next = prepare(cfg, asset, cfg.features)
    r, L = series["r"], cfg.stress_length
    rows = []
    for fold in folds:
        Fz = fold.scale(F)
        fitted = {}
        for name in cfg.baselines:
            if name == "prior":
                continue
            m = ModelRegistry.create(name)
            thr, *_ = fit_eval(m, Fz, r, fold, err_next)
            fitted[name] = (m, thr)
        if use_nets:
            for net in cfg.nets:
                m = ModelRegistry.create(net, len(cfg.features), tuned[net], cfg.seeds[0])
                thr, *_ = fit_eval(m, Fz, r, fold, err_next)
                fitted[net] = (m, thr)
        stored = {}
        for s in main["stress"]:
            if s["fold"] == fold.index:
                stored[(s["method"], s["scenario"], s["start"])] = s["delay"]
        starts = []
        for s in main["stress"]:
            if s["fold"] == fold.index and s["start"] not in starts:
                starts.append(s["start"])
        # исходный ряд: оценки по всему тестовому блоку считаются один раз
        base_alarm = {}
        for name, (m, thr) in fitted.items():
            base_alarm[name] = m.score(Fz, r, fold.test) >= thr
        test_days = fold.test + 1                                  # образец k даёт решение на день k+1
        for start in starts:
            for scen_name, Scen in SCENARIOS.items():
                mod = AnomalyInjector.apply(series, Scen(start, L, cfg.stress_factor))
                Fz_mod = fold.scale(fb.build(mod))
                idx = np.arange(start - 1, start + L + 1)
                for name, (m, thr) in fitted.items():
                    alarms = m.score(Fz_mod, mod["r"], idx) >= thr
                    d_mod = LatencyAnalyzer.delay(idx + 1, alarms, start, L)
                    rows.append({"method": name, "fold": fold.index, "scenario": scen_name, "start": int(start),
                                 "delay": d_mod, "delay_stored": stored.get((name, scen_name, start), "absent"),
                                 "share": _alarm_share(idx + 1, alarms, start, L)})
            for name in fitted:
                a = base_alarm[name]
                rows.append({"method": name, "fold": fold.index, "scenario": "control", "start": int(start),
                             "delay": LatencyAnalyzer.delay(test_days, a, start, L), "delay_stored": None,
                             "share": _alarm_share(test_days, a, start, L)})
        print(f"{asset}: блок {fold.index + 1}/{len(folds)}", flush=True)
    return {"asset": asset, "rows": rows}


def summarize(rows: list, L: int) -> dict:
    out = {}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        o = {}
        for scen in ("shock", "drift", "control"):
            rr = [r for r in rows if r["method"] == m and r["scenario"] == scen]
            d = [r["delay"] for r in rr]
            o[scen] = {**LatencyAnalyzer.summarize(d, L), "alarm_share_window": float(np.mean([r["share"] for r in rr]))}
            if scen != "control":
                chk = [r for r in rr if r["delay_stored"] != "absent"]
                o[scen]["reproduced"] = float(np.mean([r["delay"] == r["delay_stored"] for r in chk])) if chk else None
        # парная разность задержек «контроль минус сценарий» (пропуск = L + 1)
        key = lambda r: (r["fold"], r["start"], r.get("asset"))
        ctrl = {key(r): (r["delay"] or L + 1) for r in rows if r["method"] == m and r["scenario"] == "control"}
        for scen in ("shock", "drift"):
            diff = [ctrl[key(r)] - (r["delay"] or L + 1) for r in rows
                    if r["method"] == m and r["scenario"] == scen and key(r) in ctrl]
            o[scen]["speedup_mean"] = float(np.mean(diff))
            o[scen]["faster_share"] = float(np.mean([x > 0 for x in diff]))
            o[scen]["same_share"] = float(np.mean([x == 0 for x in diff]))
        out[m] = o
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, min(5, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--no-nets", action="store_true", help="только эталонные методы (без PyTorch)")
    a = ap.parse_args()
    cfg = ExperimentConfig()
    tasks = [(x, not a.no_nets) for x in cfg.assets]
    with ProcessPoolExecutor(max_workers=min(a.jobs, len(tasks))) as pool:
        res = list(pool.map(control_asset, tasks))
    rows = [{**r, "asset": x["asset"]} for x in res for r in x["rows"]]
    summ = summarize(rows, cfg.stress_length)
    name = "stress_control" + ("_baselines" if a.no_nets else "")
    (RESULTS / f"{name}.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"\n{'метод':<24}{'шок':>7}{'контр.':>8}{'ускор.':>8}{'дрейф':>7}{'ускор.':>8}"
          f"{'тревог: шок':>12}{'контр.':>8}{'1 день: шок':>12}{'контр.':>8}{'воспр.':>8}")
    for m, o in sorted(summ.items(), key=lambda kv: kv[1]["shock"]["mean_delay_censored"]):
        s, d, c = o["shock"], o["drift"], o["control"]
        print(f"{TITLES.get(m, m):<24}{s['mean_delay_censored']:>7.2f}{c['mean_delay_censored']:>8.2f}"
              f"{s['speedup_mean']:>8.2f}{d['mean_delay_censored']:>7.2f}{d['speedup_mean']:>8.2f}"
              f"{s['alarm_share_window']:>12.2f}{c['alarm_share_window']:>8.2f}{s['within_1']:>12.2f}"
              f"{c['within_1']:>8.2f}{(s['reproduced'] or 0):>8.2f}")
    print(f"\nзаписано: results/{name}.json")


if __name__ == "__main__":
    main()
