"""Автоматические проверки: отсутствие утечки данных, корректность окон, меток, эталонов, метрик и сетей."""
import numpy as np
import pytest

from fqv.core.config import ExperimentConfig, NetConfig
from fqv.data import MarketDataLoader, FeatureBuilder, DatasetBuilder, AnomalyInjector, LocalShock, StructuralDrift
from fqv.evaluation import MetricsCalculator, LatencyAnalyzer, bootstrap_ci, holm, friedman_ranks
from fqv.models import ModelRegistry, MovingAverageDetector, EwmaDetector, GarchDetector


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    df = MarketDataLoader(str(d), synthetic=True).load("TEST")
    series = FeatureBuilder.base_series(df)
    fb = FeatureBuilder()
    F = fb.build(series)
    db = DatasetBuilder()
    return series, fb, F, db, db.folds(F, series["r"]), db.error_next(series["r"])


def test_features_are_causal(data):
    """Изменение будущих данных не меняет признаки прошлых дней."""
    series, fb, F, *_ = data
    t = 1500
    mod = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in series.items()}
    mod["r"][t + 1:] *= 5
    mod["volume"][t + 1:] *= 7
    mod["hl"][t + 1:] *= 3
    assert np.allclose(fb.build(mod)[:t + 1], F[:t + 1], equal_nan=True)


def test_no_nan_in_sample_windows(data):
    series, fb, F, db, folds, _ = data
    days = db.sample_days(len(series["r"]))
    assert not np.isnan(db.windows(F, days, db.max_window)).any()


def test_folds_are_ordered_and_disjoint(data):
    *_, folds, _ = data
    for f in folds:
        assert f.train[-1] < f.val[0] and f.val[-1] < f.test[0]
        assert f.test[0] - f.val[-1] >= 2          # пропуск между обучением и тестом
    for a, b in zip(folds, folds[1:]):
        assert a.test[-1] < b.test[0]


def test_threshold_and_scaler_use_train_only(data):
    """Порог метки и нормировка не меняются, если изменить данные после обучающей части."""
    series, fb, F, db, folds, _ = data
    f0 = folds[0]
    mod = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in series.items()}
    mod["r"][f0.train[-1] + 2:] *= 10
    f0m = db.folds(fb.build(mod), mod["r"])[0]
    assert f0m.threshold == pytest.approx(f0.threshold)
    assert np.allclose(f0m.mean, f0.mean) and np.allclose(f0m.std, f0.std)


def test_label_is_next_day_error(data):
    series, _, _, db, folds, err_next = data
    k = folds[0].test[5]
    assert err_next[k] == pytest.approx(abs(series["r"][k + 1]))
    y = folds[0].labels(err_next, folds[0].train)
    assert 0.2 < y.mean() < 0.3                     # квантиль 0,75 даёт около четверти аномалий


def test_window_ends_at_sample_day(data):
    _, _, F, db, folds, _ = data
    idx = folds[0].test[:3]
    w = db.windows(F, idx, 10)
    assert w.shape == (3, 10, F.shape[1])
    assert np.allclose(w[:, -1, :], F[idx]) and np.allclose(w[:, 0, :], F[idx - 9])


@pytest.mark.parametrize("cls", [MovingAverageDetector, EwmaDetector])
def test_score_baselines_do_not_see_target_day(data, cls):
    """Оценка для дня k не зависит от доходности дня k+1 (ошибка отчёта за 6 семестр)."""
    series, *_ = data
    r = series["r"].copy()
    idx = np.array([800, 801, 1200])
    s1 = cls().score(None, r, idx)
    r2 = r.copy()
    r2[idx + 1] += 0.5
    s2 = cls().score(None, r2, idx)
    assert s1[0] == pytest.approx(s2[0]) and s1[2] == pytest.approx(s2[2])
    assert s2[1] != pytest.approx(s1[1])            # день 801 – это k+1 для образца 800, для образца 801 он «сегодня»


def test_moving_average_value():
    r = np.arange(1.0, 21.0)
    assert MovingAverageDetector(7).score(None, r, np.array([10]))[0] == pytest.approx(np.mean(r[4:11]))


def test_garch_fit_and_forecast(data):
    series, _, F, _, folds, err_next = data
    g = GarchDetector().fit(F, series["r"], folds[0], None, None)
    assert 0 < g.alpha < 0.5 and 0 < g.beta < 1 and g.alpha + g.beta < 1
    s = g.score(F, series["r"], folds[0].test)
    assert np.all(s > 0) and np.corrcoef(s, err_next[folds[0].test])[0, 1] > 0


def test_injection_changes_only_segment(data):
    series, *_ = data
    for scen in (LocalShock(1000, 10, 3.0), StructuralDrift(1000, 10, 3.0)):
        mod = AnomalyInjector.apply(series, scen)
        changed = np.where(mod["r"] != series["r"])[0]
        assert changed.min() >= 1000 and changed.max() <= 1009
        assert np.array_equal(series["r"][:1000], mod["r"][:1000])


def test_latency():
    days = np.arange(100, 112)
    alarms = np.zeros(12, bool)
    assert LatencyAnalyzer.delay(days, alarms, 100, 10) is None
    alarms[0] = True                                 # тревога на день начала шока не считается
    assert LatencyAnalyzer.delay(days, alarms, 100, 10) is None
    alarms[3] = True
    assert LatencyAnalyzer.delay(days, alarms, 100, 10) == 3
    s = LatencyAnalyzer.summarize([1, 3, None, 2], 10)
    assert s["detected_share"] == 0.75 and s["within_1"] == 0.25 and s["mean_delay_censored"] == pytest.approx(4.25)


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    thr = MetricsCalculator.best_threshold(y, s)
    assert thr == pytest.approx(0.35)
    m = MetricsCalculator.compute(y, s, thr)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 0, 2)
    assert m["recall"] == 1.0 and m["precision"] == 0.75 and m["f1"] == pytest.approx(6 / 7)
    const = MetricsCalculator.compute(y, np.full(6, 0.5), 0.5)
    assert const["roc_auc"] == 0.5 and const["pr_auc"] == pytest.approx(0.5)


def test_statistics():
    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), B=2000)
    assert lo < 3 < hi
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    fr = friedman_ranks({"good": [0.9, 0.8, 0.85, 0.95], "mid": [0.7, 0.6, 0.65, 0.7], "bad": [0.5, 0.4, 0.3, 0.2]})
    assert fr["mean_rank"]["good"] == 1.0 and fr["mean_rank"]["bad"] == 3.0 and fr["p"] < 0.05


def test_registry_and_baselines_run(data):
    series, _, F, _, folds, err_next = data
    f = folds[0]
    Fz = f.scale(F)
    y_tr, y_va = f.labels(err_next, f.train), f.labels(err_next, f.val)
    for name in ExperimentConfig().baselines:
        m = ModelRegistry.create(name).fit(Fz, series["r"], f, y_tr, y_va)
        s = m.score(Fz, series["r"], f.test)
        assert s.shape == (len(f.test),) and np.all(np.isfinite(s))
    with pytest.raises(KeyError):
        ModelRegistry.create("unknown")


# ---------------------------------------------------------------- проверки сетей (нужен PyTorch)
@pytest.mark.parametrize("kind", ["cnn", "gru", "hybrid"])
@pytest.mark.parametrize("window", [10, 30])
def test_network_shapes_and_training(data, kind, window):
    torch = pytest.importorskip("torch")
    series, _, F, _, folds, err_next = data
    f = folds[0]
    Fz = f.scale(F)
    cfg = NetConfig(window=window, hidden=8, max_epochs=3, patience=2)
    m = ModelRegistry.create(kind, F.shape[1], cfg, seed=0)
    out = m.net(torch.zeros(4, window, F.shape[1]))
    assert out.shape == (4,)
    m.fit(Fz, series["r"], f, f.labels(err_next, f.train), f.labels(err_next, f.val))
    s = m.score(Fz, series["r"], f.test)
    assert s.shape == (len(f.test),) and np.all((s >= 0) & (s <= 1))
    assert 1 <= m.session.best_epoch <= 3 and m.n_params > 0
    assert m.saliency(Fz, f.test[:16]).shape == (window, F.shape[1])


def test_network_is_reproducible(data):
    pytest.importorskip("torch")
    series, _, F, _, folds, err_next = data
    f = folds[0]
    Fz = f.scale(F)
    cfg = NetConfig(window=10, hidden=8, max_epochs=2, patience=2)
    runs = []
    for _ in range(2):
        m = ModelRegistry.create("gru", F.shape[1], cfg, seed=3)
        m.fit(Fz, series["r"], f, f.labels(err_next, f.train), f.labels(err_next, f.val))
        runs.append(m.score(Fz, series["r"], f.test[:50]))
    assert np.allclose(runs[0], runs[1], atol=1e-6)
