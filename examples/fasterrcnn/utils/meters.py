"""Minimal meter shims replacing the historical ``torchnet.meter`` dependency.

Only the interfaces used by ``trainer.FasterRCNNTrainer`` are implemented:
``AverageValueMeter`` (add/reset/value) for loss averaging and a no-frills
``ConfusionMeter`` (add/reset/value) for RPN/ROI confusion tracking.
"""

from __future__ import annotations

import numpy as np
import torch


class AverageValueMeter:
    def __init__(self):
        self.reset()

    def add(self, value, n: int = 1):
        if torch.is_tensor(value):
            value = float(value.detach().cpu().item())
        self.sum += value * n
        self.n += n

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def value(self):
        mean = self.sum / self.n if self.n else float("nan")
        return mean, float("nan")


class ConfusionMeter:
    def __init__(self, k: int, normalized: bool = False):
        self.k = k
        self.normalized = normalized
        self.conf = np.zeros((k, k), dtype=np.int64)

    def add(self, predicted, target):
        if torch.is_tensor(predicted):
            predicted = predicted.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        predicted = np.asarray(predicted)
        target = np.asarray(target).astype(np.int64).reshape(-1)
        if predicted.ndim == 2:
            predicted = predicted.argmax(axis=1)
        predicted = predicted.astype(np.int64).reshape(-1)
        valid = (target >= 0) & (target < self.k)
        for p, t in zip(predicted[valid], target[valid]):
            self.conf[t, p] += 1

    def reset(self):
        self.conf.fill(0)

    def value(self):
        if self.normalized:
            row_sums = self.conf.sum(axis=1, keepdims=True).clip(min=1)
            return self.conf / row_sums
        return self.conf
