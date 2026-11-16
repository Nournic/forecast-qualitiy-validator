"""Конфигурация эксперимента. Все параметры собраны в одном месте и сохраняются вместе с результатами."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class NetConfig:
    """Гиперпараметры нейросетевого валидатора."""
    window: int = 30          # длина входного окна, дней
    hidden: int = 32          # число фильтров свёртки / размер скрытого состояния GRU
    kernel: int = 3           # размер ядра свёртки
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8         # ранняя остановка: эпох без улучшения на валидации

    def with_(self, **kw) -> "NetConfig":
        return replace(self, **kw)

    def key(self) -> str:
        return f"w{self.window}_h{self.hidden}_k{self.kernel}_d{self.dropout}_lr{self.lr}"


@dataclass
class ExperimentConfig:
    assets: tuple = ("BTC-USD", "ETH-USD", "^GSPC", "AAPL", "TSLA")
    start: str = "2018-01-01"
    end: str = "2026-11-01"
    features: tuple = ("ret", "vol7", "vol30", "dvol", "range")
    label_quantile: float = 0.75      # порог ошибки наивного прогноза: квантиль по обучающей части фолда
    max_window: int = 60              # все конфигурации оцениваются на одних и тех же целевых днях
    warmup: int = 30                  # дней на «разогрев» скользящих признаков
    n_folds: int = 5
    initial_train: float = 0.5        # доля выборки до первого тестового блока
    val_fraction: float = 0.2         # доля валидации внутри обучающей части фолда
    seeds: tuple = (0, 1, 2, 3, 4)
    tune_seeds: tuple = (0, 1)
    ablation_seeds: tuple = (0, 1)
    nets: tuple = ("cnn", "gru", "hybrid")
    baselines: tuple = ("prior", "persist", "roll7", "ewma", "garch", "logreg", "gboost", "iforest")
    net: NetConfig = field(default_factory=NetConfig)
    # сетка «один фактор за раз» для исследования гиперпараметров
    grid: dict = field(default_factory=lambda: {
        "window": [10, 20, 30, 60], "hidden": [16, 32, 64], "dropout": [0.0, 0.2, 0.4],
        "lr": [3e-4, 1e-3, 3e-3], "kernel": [3, 5]})
    stress_injections: int = 10       # вставок шока на каждый фолд
    stress_factor: float = 3.0        # во сколько раз растёт амплитуда доходностей
    stress_length: int = 10           # длительность шока, дней
    synthetic: bool = False
    data_dir: str = "data/raw"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)

    def quick(self) -> "ExperimentConfig":
        """Сокращённый вариант для быстрой проверки работоспособности."""
        return replace(self, assets=self.assets[:2], n_folds=2, seeds=(0,), tune_seeds=(0,), ablation_seeds=(0,),
                       net=self.net.with_(max_epochs=6, patience=3),
                       grid={"window": [10, 30], "hidden": [16, 32]}, stress_injections=2)
