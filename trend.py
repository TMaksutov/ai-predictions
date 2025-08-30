from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TrendModel:
    """Simple polynomial trend model (degree <= 2) with up to 3 components.

    Components used:
    - intercept (1)
    - linear term t (2)
    - optional quadratic term t^2 (3)

    Time index t is days since the first timestamp.
    """

    t0: pd.Timestamp
    coef: np.ndarray  # shape (k,)
    degree: int

    def _to_t(self, ds: pd.Series) -> np.ndarray:
        t = (pd.to_datetime(ds, errors="coerce") - pd.Timestamp(self.t0)).dt.days.astype(float)
        return t.to_numpy()

    def _design(self, t: np.ndarray) -> np.ndarray:
        X = [np.ones_like(t)]
        if self.degree >= 1:
            X.append(t)
        if self.degree >= 2:
            X.append(t ** 2)
        return np.vstack(X).T

    def fitted(self, ds: pd.Series) -> pd.Series:
        t = self._to_t(ds)
        X = self._design(t)
        yhat = X @ self.coef
        return pd.Series(yhat, index=pd.RangeIndex(len(yhat)))

    def extrapolate(self, ds_future: pd.Series) -> pd.Series:
        return self.fitted(ds_future)


def fit_trend(series_df: pd.DataFrame, max_degree: int = 2) -> TrendModel:
    """Fit a simple polynomial trend (degree 1 or 2) to known y values.

    - Uses OLS with minimal regularization for durability.
    - Picks degree by simple holdout (last 20% of known points) RMSE.
    - Returns a single model for the whole graph.
    """
    df = series_df.copy()
    df = df.sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).reset_index(drop=True)
    if len(df) < 5:
        # Degenerate: constant trend
        return TrendModel(t0=pd.to_datetime(df["ds"].min()) if len(df) else pd.Timestamp("1970-01-01"), coef=np.array([df["y"].mean() if len(df) else 0.0]), degree=0)

    t0 = pd.to_datetime(df["ds"].min())
    t = (df["ds"] - t0).dt.days.astype(float).to_numpy()
    y = df["y"].to_numpy(dtype=float)

    # Simple split
    n = len(df)
    split = max(1, int(n * 0.8))
    t_tr, y_tr = t[:split], y[:split]
    t_te, y_te = t[split:], y[split:]
    if len(t_te) == 0:
        t_tr, y_tr = t, y
        t_te, y_te = t[-1:], y[-1:]

    best_degree = 0
    best_coef = None
    best_rmse = float("inf")

    for deg in range(0, max_degree + 1):
        # Build design
        X_tr = [np.ones_like(t_tr)]
        if deg >= 1:
            X_tr.append(t_tr)
        if deg >= 2:
            X_tr.append(t_tr ** 2)
        X_tr = np.vstack(X_tr).T
        # Ridge-like tiny regularization for stability
        lam = 1e-6
        XtX = X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1])
        Xty = X_tr.T @ y_tr
        coef = np.linalg.solve(XtX, Xty)

        # Validate
        X_te = [np.ones_like(t_te)]
        if deg >= 1:
            X_te.append(t_te)
        if deg >= 2:
            X_te.append(t_te ** 2)
        X_te = np.vstack(X_te).T
        yhat_te = X_te @ coef
        rmse = float(np.sqrt(np.mean((y_te - yhat_te) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_degree = deg
            best_coef = coef

    return TrendModel(t0=t0, coef=np.asarray(best_coef), degree=int(best_degree))


