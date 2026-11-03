"""
Modeli linear "FedAvg-compatible" per çdo klient.

Pse linear (jo Random Forest/XGBoost si ne punimin e meparshem): FedAvg
(mesatarizim peshash) ka kuptim vetem nese te gjithe klientet trajnojne TE
NJEJTIN arkitekture parametrash dhe mund te pervetesojne ne menyre inkrementale
(epoch-by-epoch) -- dy veti qe pemet (RF/XGBoost) nuk i kane, por qe perputhen
natyrshem me regresion linear te trajnuar me gradient descent (njesoj si
modeli "lite" qe eksportohet per Arduino MKR1000 ne punimin e meparshem).
Mbajme Persistence si baseline (jo-FL, per referim).

Dy varianta parametrash:
  - "full"  -- te gjitha features e inxhinierizuara (lag/rolling/ciklike).
  - "lite"  -- vetem vlerat aktuale te 2 sensoreve (temp, humidity), njesoj
    si modeli i eksportuar per mikrokontrollues -- perdoret kur kontrolluesi
    adaptiv vendos "kalo ne model me te vogel" (vendimi #6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LinearFedModel:
    n_features: int
    weights: np.ndarray = field(default=None)  # shape (n_features + 1,), elementi i fundit = bias
    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.zeros(self.n_features + 1, dtype=np.float64)

    def set_scaler(self, X: np.ndarray):
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None:
            self.set_scaler(X)
        return (X - self.feature_mean) / self.feature_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(X)
        return Xs @ self.weights[:-1] + self.weights[-1]

    def local_train(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float = 0.01,
                     l2: float = 1e-2, batch_size: int = 64, max_grad_norm: float = 5.0,
                     rng: Optional[np.random.Generator] = None):
        """Gradient descent per 'epochs' kalime te plota -- kjo eshte PIKERISHT
        dimensioni 'local epochs' qe kontrollon kontrolluesi adaptiv (vendimi #2).

        Features e inxhinierizuara (lags/rolling) jane shume te korreluara, qe
        e ben gradient descent te thjeshte te paqendrueshme me lr te larta --
        prandaj lr e ulet + L2 + gradient clipping jane parametra parazgjedhur
        te sigurt, jo thjesht kozmetike."""
        if rng is None:
            rng = np.random.default_rng(0)
        if self.feature_mean is None:
            self.set_scaler(X)
        Xs = self._standardize(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(y)
        if n == 0 or epochs <= 0:
            return
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                Xb, yb = Xs[b], y[b]
                pred = Xb @ self.weights[:-1] + self.weights[-1]
                err = pred - yb
                grad_w = (Xb.T @ err) / len(b) + l2 * self.weights[:-1]
                grad_b = err.mean()
                grad_norm = float(np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2))
                if grad_norm > max_grad_norm:
                    factor = max_grad_norm / (grad_norm + 1e-12)
                    grad_w = grad_w * factor
                    grad_b = grad_b * factor
                self.weights[:-1] -= lr * grad_w
                self.weights[-1] -= lr * grad_b
                if not np.all(np.isfinite(self.weights)):
                    # siguri e fundit: mos lejo kurre peshat te shperthejne ne NaN/inf
                    self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def set_weights(self, w: np.ndarray):
        self.weights = w.copy()

    def num_bytes_full(self) -> int:
        return self.weights.nbytes  # float64


def persistence_predict(y_last: np.ndarray) -> np.ndarray:
    """Baseline naive: y(t+h) = y(t)."""
    return y_last


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill_vs_persistence(rmse_model: float, rmse_persist: float) -> float:
    if rmse_persist == 0:
        return 0.0
    return 100.0 * (rmse_persist - rmse_model) / rmse_persist
