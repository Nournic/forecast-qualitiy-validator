"""ExperimentRunner – фасад комплекса: подбор гиперпараметров, основной эксперимент, абляция признаков.
Каждый актив обрабатывается в отдельном процессе; результаты по активам сохраняются сразу, поэтому прерванный
прогон можно продолжить той же командой."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from ..data import MarketDataLoader, FeatureBuilder, DatasetBuilder, AnomalyInjector, LocalShock, StructuralDrift
from ..evaluation.latency import LatencyAnalyzer
from ..evaluation.metrics import MetricsCalculator
from ..models.registry import ModelRegistry, NET_NAMES
from .config import ExperimentConfig, NetConfig
from .store import ResultStore
from .utils import environment_info

FEATURE_SETS = {"R": ("ret",), "RV": ("ret", "vol7", "vol30")}


def log(msg: str, results_dir: str = "results") -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(f"{results_dir}/run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def is_crypto(asset: str) -> bool:
    return asset.endswith("-USD")


def prepare(cfg: ExperimentConfig, asset: str, features: tuple):
    """Загрузка данных актива, построение признаков и разбиений."""
    loader = MarketDataLoader(cfg.data_dir, cfg.start, cfg.end, cfg.synthetic)
    df = loader.load(asset)
    fng = loader.load_fear_greed() if "fng" in features else None
    if "fng" in features and fng is None:
        return None
    series = FeatureBuilder.base_series(df, fng)
    if "fng" in series:
        series["fng"] = np.where(np.isnan(series["fng"]), 0.5, series["fng"])
    fb = FeatureBuilder(features)
    F = fb.build(series)
    db = DatasetBuilder(cfg.n_folds, cfg.initial_train, cfg.val_fraction, cfg.label_quantile, cfg.max_window,
                        cfg.warmup)
    return series, fb, F, db, db.folds(F, series["r"]), db.error_next(series["r"])


def fit_eval(model, Fz, r, fold, err_next) -> tuple:
    """Обучение на фолде, выбор порога по валидации, метрики на валидации и тесте."""
    y_tr, y_va, y_te = (fold.labels(err_next, i) for i in (fold.train, fold.val, fold.test))
    t0 = time.perf_counter()
    model.fit(Fz, r, fold, y_tr, y_va)
    fit_time = time.perf_counter() - t0
    s_va = model.score(Fz, r, fold.val)
    thr = MetricsCalculator.best_threshold(y_va, s_va)
    val = MetricsCalculator.compute(y_va, s_va, thr, model.probabilistic)
    test = MetricsCalculator.compute(y_te, model.score(Fz, r, fold.test), thr, model.probabilistic)
    return thr, val, test, fit_time


def ofat_configs(cfg: ExperimentConfig) -> list:
    """Конфигурации «один фактор за раз»: базовая и отличающиеся от неё одним гиперпараметром."""
    base, out = cfg.net, [cfg.net]
    for factor, values in cfg.grid.items():
        for v in values:
            if getattr(base, factor) != v:
                out.append(base.with_(**{factor: v}))
    return out


def _torch_single_thread():
    import torch
    torch.set_num_threads(1)


# ---------------------------------------------------------------- этап 1: гиперпараметры
def tune_asset(args) -> dict:
    cfg, asset = args
    _torch_single_thread()
    series, fb, F, db, folds, err_next = prepare(cfg, asset, cfg.features)
    fold = folds[0]                               # только данные до первого тестового блока
    Fz, r = fold.scale(F), series["r"]
    rows = []
    for nc in ofat_configs(cfg):
        for net in cfg.nets:
            for seed in cfg.tune_seeds:
                m = ModelRegistry.create(net, len(cfg.features), nc, seed)
                _, val, _, t = fit_eval(m, Fz, r, replace(fold, test=fold.val), err_next)
                rows.append({"asset": asset, "net": net, "seed": seed, "config": nc.key(), "window": nc.window,
                             "hidden": nc.hidden, "dropout": nc.dropout, "lr": nc.lr, "kernel": nc.kernel,
                             "val_pr_auc": val["pr_auc"], "val_roc_auc": val["roc_auc"], "val_f1": val["f1"],
                             "epochs": m.session.best_epoch, "time": t, "n_params": m.n_params})
    log(f"подбор гиперпараметров: {asset} готов ({len(rows)} обучений)", cfg.results_dir)
    return {"asset": asset, "rows": rows}


def select_configs(cfg: ExperimentConfig, tuning: list) -> dict:
    """Для каждой сети – конфигурация с наибольшей средней PR-AUC на валидации (по активам и запускам)."""
    by_key = {nc.key(): nc for nc in ofat_configs(cfg)}
    best = {}
    for net in cfg.nets:
        score = {}
        for t in tuning:
            for row in t["rows"]:
                if row["net"] == net:
                    score.setdefault(row["config"], []).append(row["val_pr_auc"])
        key = max(score, key=lambda k: np.nanmean(score[k]))
        best[net] = by_key[key]
    return best


# ---------------------------------------------------------------- этап 2: основной эксперимент
def main_asset(args) -> dict:
    cfg, asset, tuned = args
    _torch_single_thread()
    series, fb, F, db, folds, err_next = prepare(cfg, asset, cfg.features)
    r, dates, C = series["r"], series["dates"], len(cfg.features)
    rng = np.random.default_rng(12345)
    out = {"asset": asset, "n_days": int(len(r)), "first_day": str(dates[0].date()), "last_day": str(dates[-1].date()),
           "folds": [], "records": [], "inference": {}, "importance": [], "saliency": [], "stress": [],
           "history": {}}
    for fold in folds:
        Fz = fold.scale(F)
        y_te = fold.labels(err_next, fold.test)
        out["folds"].append({"fold": fold.index, "n_train": len(fold.train), "n_val": len(fold.val),
                             "n_test": len(fold.test), "threshold": fold.threshold,
                             "test_from": str(dates[fold.test[0] + 1].date()),
                             "test_to": str(dates[fold.test[-1] + 1].date()),
                             "prevalence_train": float(fold.labels(err_next, fold.train).mean()),
                             "prevalence_test": float(y_te.mean())})
        fitted = {}                                   # модели и пороги для стресс-теста
        for name in cfg.baselines:
            m = ModelRegistry.create(name)
            thr, val, test, t = fit_eval(m, Fz, r, fold, err_next)
            out["records"].append({"method": name, "fold": fold.index, "seed": None, "threshold": thr,
                                   "fit_time": t, "val": val, "test": test})
            fitted[name] = (m, thr)
        for net in cfg.nets:
            for seed in cfg.seeds:
                m = ModelRegistry.create(net, C, tuned[net], seed)
                thr, val, test, t = fit_eval(m, Fz, r, fold, err_next)
                out["records"].append({"method": net, "fold": fold.index, "seed": seed, "threshold": thr,
                                       "fit_time": t, "epochs": m.session.best_epoch,
                                       "epochs_run": len(m.session.history["val_loss"]), "n_params": m.n_params,
                                       "val": val, "test": test})
                if seed == cfg.seeds[0]:
                    fitted[net] = (m, thr)
                    from ..evaluation.importance import permutation_importance
                    out["importance"].append({"method": net, "fold": fold.index, "features": list(cfg.features),
                                              "drop_pr_auc": permutation_importance(m, Fz, r, fold.test, y_te, C)})
                    out["saliency"].append({"method": net, "fold": fold.index,
                                            "by_lag": m.saliency(Fz, fold.test).sum(axis=1).tolist()})
                    if fold.index == 0:
                        out["inference"][net] = m.inference_time(Fz, fold.test)
                    if fold.index == len(folds) - 1:
                        out["history"][net] = m.session.history
        out["stress"].extend(stress_fold(cfg, series, fb, fold, fitted, rng))
        log(f"основной эксперимент: {asset}, фолд {fold.index + 1}/{len(folds)}", cfg.results_dir)
    return out


def stress_fold(cfg: ExperimentConfig, series: dict, fb: FeatureBuilder, fold, fitted: dict, rng) -> list:
    """Вставка шока в тестовую часть фолда и измерение задержки первой тревоги для каждого метода."""
    L, rows = cfg.stress_length, []
    lo, hi = fold.test[0] + 6, fold.test[-1] - L - 2          # день начала шока (целевой день)
    if hi <= lo:
        return rows
    for i in range(cfg.stress_injections):
        start = int(rng.integers(lo, hi))
        for scen_name, scen in (("shock", LocalShock(start, L, cfg.stress_factor)),
                                ("drift", StructuralDrift(start, L, cfg.stress_factor))):
            mod = AnomalyInjector.apply(series, scen)
            Fz_mod = fold.scale(fb.build(mod))
            idx = np.arange(start - 1, start + L + 1)           # образец k даёт решение на день k+1
            for name, (m, thr) in fitted.items():
                if name == "prior":
                    continue
                alarms = m.score(Fz_mod, mod["r"], idx) >= thr
                rows.append({"method": name, "fold": fold.index, "scenario": scen_name, "start": start,
                             "delay": LatencyAnalyzer.delay(idx + 1, alarms, start, L)})
    return rows


# ---------------------------------------------------------------- этап 3: абляция признаков
def ablation_asset(args) -> dict:
    cfg, asset, tuned = args
    _torch_single_thread()
    sets = dict(FEATURE_SETS)
    if is_crypto(asset):
        sets["full+fng"] = tuple(cfg.features) + ("fng",)
    rows = []
    for set_name, feats in sets.items():
        prep = prepare(cfg, asset, feats)
        if prep is None:
            continue
        series, fb, F, db, folds, err_next = prep
        for fold in folds:
            Fz, r = fold.scale(F), series["r"]
            m = ModelRegistry.create("logreg")
            _, _, test, _ = fit_eval(m, Fz, r, fold, err_next)
            rows.append({"asset": asset, "set": set_name, "method": "logreg", "fold": fold.index, "seed": None,
                         "test": test})
            for net in cfg.nets:
                for seed in cfg.ablation_seeds:
                    m = ModelRegistry.create(net, len(feats), tuned[net], seed)
                    _, _, test, _ = fit_eval(m, Fz, r, fold, err_next)
                    rows.append({"asset": asset, "set": set_name, "method": net, "fold": fold.index, "seed": seed,
                                 "test": test})
        log(f"абляция признаков: {asset}, набор {set_name}", cfg.results_dir)
    return {"asset": asset, "rows": rows}


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, jobs: int = 1, force: bool = False):
        self.cfg, self.jobs, self.force = cfg, max(1, jobs), force
        self.store = ResultStore(cfg.results_dir)

    def _map(self, fn, tasks: list, prefix: str) -> list:
        """Выполняет задачи по активам, пропуская уже посчитанные (если не задан force)."""
        todo = [t for t in tasks if self.force or not self.store.has(f"{prefix}_{_safe(t[1])}")]
        if todo:
            if self.jobs == 1 or len(todo) == 1:
                done = map(fn, todo)
            else:
                pool = ProcessPoolExecutor(max_workers=min(self.jobs, len(todo)))
                done = pool.map(fn, todo)
            for res in done:
                self.store.save(f"{prefix}_{_safe(res['asset'])}", res)
        return [self.store.load(f"{prefix}_{_safe(t[1])}") for t in tasks]

    def download(self) -> None:
        loader = MarketDataLoader(self.cfg.data_dir, self.cfg.start, self.cfg.end, self.cfg.synthetic)
        for a in self.cfg.assets:
            df = loader.load(a)
            log(f"данные: {a}, {len(df)} дней, {df.index[0].date()} – {df.index[-1].date()}", self.cfg.results_dir)
        fng = loader.load_fear_greed()
        log("индекс страха и жадности: " + (f"{len(fng)} дней" if fng is not None else "недоступен"),
            self.cfg.results_dir)

    def run(self) -> dict:
        cfg, t0 = self.cfg, time.time()
        self.store.save("config", cfg.to_dict())
        self.store.save("env", environment_info())
        self.download()
        tuning = self._map(tune_asset, [(cfg, a) for a in cfg.assets], "tune")
        tuned = select_configs(cfg, tuning)
        self.store.save("tuned", {k: v.__dict__ for k, v in tuned.items()})
        log("выбранные конфигурации: " + ", ".join(f"{k}: {v.key()}" for k, v in tuned.items()), cfg.results_dir)
        main = self._map(main_asset, [(cfg, a, tuned) for a in cfg.assets], "main")
        ablation = self._map(ablation_asset, [(cfg, a, tuned) for a in cfg.assets], "ablation")
        from ..evaluation.report import ComparisonReport
        summary = ComparisonReport(cfg, tuning, main, ablation).build()
        summary["elapsed_minutes"] = (time.time() - t0) / 60
        self.store.save("summary", summary)
        log(f"готово за {summary['elapsed_minutes']:.1f} мин", cfg.results_dir)
        return summary


def _safe(asset: str) -> str:
    return asset.replace("^", "").replace("=", "_")
