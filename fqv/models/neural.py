"""Нейросетевые валидаторы: 1D-CNN, GRU и гибрид CNN+GRU. NeuralValidator объединяет сеть и TrainingSession."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from ..core.config import NetConfig
from ..data.dataset import DatasetBuilder
from .base import AnomalyValidator


class CNN1D(nn.Module):
    """Два свёрточных блока, глобальный максимум по времени, линейный классификатор."""

    def __init__(self, n_features: int, cfg: NetConfig):
        super().__init__()
        p = cfg.kernel // 2
        self.conv1 = nn.Conv1d(n_features, cfg.hidden, cfg.kernel, padding=p)
        self.conv2 = nn.Conv1d(cfg.hidden, 2 * cfg.hidden, cfg.kernel, padding=p)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(2 * cfg.hidden, 1)

    def forward(self, x):                       # x: (батч, окно, признаки)
        h = x.permute(0, 2, 1)
        h = self.pool(torch.relu(self.conv1(h)))
        h = torch.relu(self.conv2(h))
        h = torch.amax(h, dim=2)
        return self.fc(self.drop(h)).squeeze(1)


class GRUNet(nn.Module):
    """Один слой GRU, классификация по последнему скрытому состоянию."""

    def __init__(self, n_features: int, cfg: NetConfig):
        super().__init__()
        self.gru = nn.GRU(n_features, cfg.hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(self.drop(h[-1])).squeeze(1)


class HybridCNNGRU(nn.Module):
    """Свёртка выделяет локальные признаки, GRU обрабатывает их последовательность."""

    def __init__(self, n_features: int, cfg: NetConfig):
        super().__init__()
        self.conv = nn.Conv1d(n_features, cfg.hidden, cfg.kernel, padding=cfg.kernel // 2)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden, 1)

    def forward(self, x):
        h = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        _, h = self.gru(h)
        return self.fc(self.drop(h[-1])).squeeze(1)


NETWORKS = {"cnn": CNN1D, "gru": GRUNet, "hybrid": HybridCNNGRU}


class TrainingSession:
    """Обучение с ранней остановкой по функции потерь на валидации и снижением скорости обучения."""

    def __init__(self, net: nn.Module, cfg: NetConfig, seed: int):
        self.net, self.cfg, self.seed = net, cfg, seed
        self.history = {"train_loss": [], "val_loss": []}
        self.best_epoch, self.train_time = 0, 0.0

    def run(self, x_tr, y_tr, x_va, y_va) -> None:
        cfg = self.cfg
        gen = torch.Generator().manual_seed(self.seed)
        pos = float(y_tr.sum())
        pos_weight = torch.tensor((len(y_tr) - pos) / max(pos, 1.0))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        best, best_state, bad = float("inf"), None, 0
        t0 = time.perf_counter()
        n = len(x_tr)
        for epoch in range(1, cfg.max_epochs + 1):
            self.net.train()
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for i in range(0, n, cfg.batch_size):
                b = perm[i:i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(x_tr[b]), y_tr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(x_va), y_va).item()
            self.history["train_loss"].append(total / n)
            self.history["val_loss"].append(val)
            sched.step(val)
            if val < best - 1e-5:
                best, bad, self.best_epoch = val, 0, epoch
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.train_time = time.perf_counter() - t0
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()


class NeuralValidator(AnomalyValidator):
    probabilistic = True

    def __init__(self, kind: str, cfg: NetConfig, n_features: int, seed: int = 0):
        self.name, self.kind, self.cfg, self.seed = kind, kind, cfg, seed
        torch.manual_seed(seed)
        self.net = NETWORKS[kind](n_features, cfg)
        self.session = TrainingSession(self.net, cfg, seed)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _tensor(self, Fz, idx):
        return torch.from_numpy(DatasetBuilder.windows(Fz, idx, self.cfg.window))

    def fit(self, Fz, r, fold, y_train, y_val):
        self.session.run(self._tensor(Fz, fold.train), torch.from_numpy(y_train),
                         self._tensor(Fz, fold.val), torch.from_numpy(y_val))
        return self

    def score(self, Fz, r, idx):
        with torch.no_grad():
            return torch.sigmoid(self.net(self._tensor(Fz, idx))).numpy().astype(np.float64)

    def inference_time(self, Fz, idx, repeats: int = 200) -> dict:
        """Среднее время одного предсказания (одно окно) и предсказания пачкой, миллисекунды."""
        x = self._tensor(Fz, idx[:256])
        one = x[:1]
        with torch.no_grad():
            for _ in range(10):
                self.net(one)
            t0 = time.perf_counter()
            for _ in range(repeats):
                self.net(one)
            t1 = time.perf_counter()
            for _ in range(20):
                self.net(x)
            t2 = time.perf_counter()
        return {"single_ms": 1000 * (t1 - t0) / repeats, "batch_ms_per_sample": 1000 * (t2 - t1) / 20 / len(x)}

    def saliency(self, Fz, idx) -> np.ndarray:
        """|градиент × вход|, усреднённый по образцам: матрица (длина окна, число признаков)."""
        x = self._tensor(Fz, idx).clone().requires_grad_(True)
        self.net(x).sum().backward()
        return (x.grad * x).abs().mean(dim=0).detach().numpy()
