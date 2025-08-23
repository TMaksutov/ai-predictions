from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    BayesianRidge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time

from features import build_features_internal, build_features

# Import configuration
try:
    from utils.config import DEFAULT_TEST_FRACTION, MIN_TRAINING_ROWS
except ImportError:
    # Fallback values
    DEFAULT_TEST_FRACTION = 0.2
    MIN_TRAINING_ROWS = 10



def _select_base_feature_columns(all_columns: List[str]) -> List[str]:
    """Pick model features we can also recreate when forecasting.

    Includes: lags, moving averages, DOW dummies, trend, Fourier terms, and exogenous features.
    """
    # Core time series features
    core_prefixes = ("lag", "ma", "dow_", "fourier_")

    # Exogenous features that can be recreated during forecasting
    exog_prefixes = ("x_", "m_")  # x_ for numeric/categorical exog, m_ for multilabel exog

    base_cols = [c for c in all_columns if
                 c.startswith(core_prefixes) or
                 c.startswith(exog_prefixes) or
                 c == "trend"]

    return base_cols


def _compute_fourier_from_name(trend_value: int, col_name: str) -> float:
    """Compute a single Fourier feature value given its column name.

    Expected format: 'fourier_p{period}_k{order}_{sin|cos}'.
    """
    try:
        # Example: fourier_p7_k3_sin
        parts = col_name.split("_")
        p_str = parts[1]  # p7
        k_str = parts[2]  # k3
        func = parts[3]   # sin or cos
        period = int(p_str[1:])
        order = int(k_str[1:])
        angle = 2.0 * np.pi * order * (float(trend_value) / float(period))
        if func == "sin":
            return float(np.sin(angle))
        return float(np.cos(angle))
    except Exception:
        return 0.0


def _build_row_for_date(
    base_cols: List[str],
    hist_df: pd.DataFrame,
    target_ds: pd.Timestamp,
    start_ds: pd.Timestamp,
) -> List[float]:
    """Create a feature row matching base_cols for a target date using history.

    - hist_df contains only known/previously predicted 'y' values with ascending 'ds'.
    - start_ds is the first timestamp of the original series for computing trend.
    """
    values = []
    trend_val = int((pd.Timestamp(target_ds) - pd.Timestamp(start_ds)).days)
    # Helpers
    y_series = hist_df["y"].astype(float)
    for col in base_cols:
        if col == "trend":
            values.append(float(trend_val))
            continue
        if col.startswith("dow_"):
            desired = int(pd.Timestamp(target_ds).dayofweek)
            current = int(col.split("_")[1])
            values.append(1.0 if current == desired else 0.0)
            continue
        if col.startswith("lag"):
            # e.g., lag7
            try:
                lag_k = int(col.replace("lag", ""))
            except Exception:
                lag_k = 1
            idx = -lag_k
            val = float(y_series.iloc[idx]) if len(y_series) >= abs(idx) else float(y_series.iloc[-1])
            values.append(val)
            continue
        if col.startswith("ma"):
            # e.g., ma14
            try:
                window = int(col.replace("ma", ""))
            except Exception:
                window = 7
            window = max(1, window)
            tail = y_series.tail(window)
            values.append(float(tail.mean()))
            continue
        if col.startswith("fourier_"):
            values.append(_compute_fourier_from_name(trend_val, col))
            continue
        if col.startswith("x_") or col.startswith("m_"):
            # Exogenous features - for forecasting without future data, use 0 as default
            # In practice, these should come from external data sources
            values.append(0.0)
            continue
        # Unknown engineered column -> default 0
        values.append(0.0)
    return values


def ensure_daily_frequency(series_df: pd.DataFrame) -> None:
    if series_df.empty:
        raise ValueError("Empty series provided")
    series_df = series_df.sort_values("ds").reset_index(drop=True)
    inferred = pd.infer_freq(series_df["ds"])
    if inferred != "D":
        raise ValueError(
            "Dataset must be daily frequency ('D') with regular 1-day intervals. "
            "Please resample/fill gaps before uploading."
        )


# Removed Linear regression compatibility wrapper


def forecast_with_estimator(series_df: pd.DataFrame, estimator, test_fraction: float = DEFAULT_TEST_FRACTION) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    if series_df.shape[0] < MIN_TRAINING_ROWS:
        raise ValueError("Insufficient data (need at least 10 rows)")
    ensure_daily_frequency(series_df)

    # Split train/test on time
    n = len(series_df)
    test_size = max(1, int(n * test_fraction))
    sorted_df = series_df.sort_values("ds").reset_index(drop=True)
    test_start_ds = sorted_df.iloc[-test_size]["ds"]

    # Build rich features and use the subset we can also reconstruct when forecasting
    features_df, feature_cols_all = build_features(sorted_df)
    base_cols = _select_base_feature_columns(feature_cols_all)

    # Align to train/test windows
    feats_train = features_df[features_df["ds"] < test_start_ds].copy()
    feats_test = features_df[features_df["ds"] >= test_start_ds].copy()

    # Handle missing values more carefully for time series
    # Only drop rows where target variable is missing, allow NaN in predictors for early periods
    if not feats_train.empty:
        feats_train = feats_train.dropna(subset=["y"]).copy()
        # Fill remaining NaN values in predictors with forward fill, then 0 for any remaining
        feats_train[base_cols] = feats_train[base_cols].fillna(method='ffill').fillna(0)

    if not feats_test.empty:
        feats_test = feats_test.dropna(subset=["y"]).copy()
        # Fill remaining NaN values in predictors with forward fill, then 0 for any remaining
        feats_test[base_cols] = feats_test[base_cols].fillna(method='ffill').fillna(0)

    # Guard against empty splits after dropna. If no test rows remain, return inf score to exclude this model.
    if feats_test.empty or ("y" not in feats_test.columns):
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0

    # If no train rows remain, propagate a clear error to be reported by the caller
    if feats_train.empty or ("y" not in feats_train.columns):
        raise ValueError("No valid training rows after feature construction; cannot fit model")

    # Convert to float arrays to handle mixed data types
    X_train = feats_train[base_cols].to_numpy(dtype=float)
    y_train = feats_train["y"].to_numpy(dtype=float)
    X_test = feats_test[base_cols].to_numpy(dtype=float)
    y_test = feats_test["y"].to_numpy(dtype=float)

    # Train and predict with timing
    model = clone(estimator)
    # Adapt K for KNN if training set is very small to avoid ValueError
    try:
        if isinstance(model, KNeighborsRegressor):
            desired_k = getattr(model, "n_neighbors", 5)
            max_allowed_k = max(1, int(min(desired_k, X_train.shape[0] if X_train is not None else 1)))
            if max_allowed_k != desired_k:
                model.set_params(n_neighbors=max_allowed_k)
    except Exception:
        pass

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time_s = float(time.time() - t0)

    t1 = time.time()
    preds = model.predict(X_test) if len(X_test) > 0 else np.array([])
    predict_time_s = float(time.time() - t1)

    forecast_df = pd.DataFrame({
        "ds": feats_test["ds"].to_numpy(),
        "yhat": preds,
    })

    if len(preds) > 0:
        se = (y_test - preds) ** 2
        mse = np.nanmean(se)
        rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("inf")
    else:
        rmse = float("inf")

    # Convert feats_test back to original structure for the app (only ds/y)
    test_df = feats_test[["ds", "y"]].reset_index(drop=True)

    return rmse, forecast_df, test_df, train_time_s, predict_time_s


class UnifiedTimeSeriesTrainer:
    """
    Unified architecture for consistent training and retraining of time series models.
    Ensures the same feature engineering, data preprocessing, and model configuration
    across all training scenarios.
    """

    def __init__(self):
        self.feature_columns: List[str] = []
        self.base_columns: List[str] = []
        self.exog_schema: List[Dict] = []
        self.trained_models: Dict[str, BaseEstimator] = {}
        self.training_history: Dict[str, Dict] = {}

    def _prepare_features(self, series_df: pd.DataFrame, return_info: bool = False) -> Tuple[pd.DataFrame, List[str], Dict]:
        """
        Unified feature engineering - used by all training functions.
        This ensures consistency between training, retraining, and forecasting.
        """
        # Build features using the same internal function
        features_df, feature_cols_all, info = build_features_internal(series_df, return_info=True)

        # Store feature information for consistency
        if not self.feature_columns:
            self.feature_columns = feature_cols_all
            self.base_columns = _select_base_feature_columns(feature_cols_all)
            self.exog_schema = info.get("exog_schema", [])

        # Expand exogenous columns
        exog_cols_expanded = []
        for sch in self.exog_schema:
            exog_cols_expanded.extend(list(sch.get("columns", [])))

        if return_info:
            return features_df, feature_cols_all, info
        return features_df, feature_cols_all

    def _get_exog_vector_for_date(self, target_ds: pd.Timestamp, future_map: Dict) -> List[float]:
        """Consistent exogenous feature vector generation"""
        values: List[float] = []
        raw = future_map.get(pd.Timestamp(target_ds), {})

        for sch in self.exog_schema:
            typ = sch.get("type")
            cols = list(sch.get("columns", []))
            if typ == "numeric":
                original_name = sch.get("original_name")
                val = pd.to_numeric(raw.get(original_name, pd.NA), errors="coerce")
                if pd.isna(val):
                    values.append(np.nan)
                else:
                    values.append(float(val))
            elif typ == "categorical":
                original_name = sch.get("original_name")
                raw_val = raw.get(original_name, pd.NA)
                raw_str = str(raw_val) if not pd.isna(raw_val) else "nan"
                hit_col = None
                for c in cols:
                    suffix = c.split(f"x_{sch.get('sanitized_base')}_", 1)[-1]
                    if raw_str == "nan" and suffix == "nan":
                        hit_col = c
                        break
                    if suffix == str(raw_val):
                        hit_col = c
                        break
                for c in cols:
                    values.append(1.0 if c == hit_col else 0.0)
            elif typ == "multilabel":
                original_name = sch.get("original_name")
                raw_val = raw.get(original_name, None)
                labels = []
                if isinstance(raw_val, str):
                    for delim in [",", ";", "|"]:
                        if delim in raw_val:
                            labels = [p.strip() for p in raw_val.split(delim) if p.strip()]
                            break
                    if not labels and raw_val.strip():
                        labels = [raw_val.strip()]
                label_set = set(labels)
                for c in cols:
                    suffix = c.split(f"m_{sch.get('sanitized_base')}_", 1)[-1]
                    values.append(1.0 if suffix in label_set else 0.0)
            else:
                for _ in cols:
                    values.append(0.0)
        return values

    def benchmark_models(self, series_df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION) -> List[Dict[str, Any]]:
        """
        Unified benchmarking that uses the same feature engineering as retraining.
        This ensures consistency between model selection and retraining.
        """
        if series_df.shape[0] < MIN_TRAINING_ROWS:
            raise ValueError("Insufficient data (need at least 10 rows)")
        ensure_daily_frequency(series_df)

        # Use unified feature preparation
        sorted_df = series_df.sort_values("ds").reset_index(drop=True)
        features_df, feature_cols_all = self._prepare_features(sorted_df)

        # Split train/test on time
        n = len(sorted_df)
        test_size = max(1, int(n * test_fraction))
        test_start_ds = sorted_df.iloc[-test_size]["ds"]

        # Align to train/test windows
        feats_train = features_df[features_df["ds"] < test_start_ds].copy()
        feats_test = features_df[features_df["ds"] >= test_start_ds].copy()

        # Drop rows with missing predictors/target
        feats_train = feats_train.dropna(subset=self.base_columns + ["y"])
        feats_test = feats_test.dropna(subset=self.base_columns + ["y"])

        if feats_test.empty:
            return [{"name": "NoModels", "rmse": float("inf"), "forecast_df": pd.DataFrame(), "test_df": pd.DataFrame()}]

        # Prepare data
        X_train = feats_train[self.base_columns].to_numpy(dtype=float)
        y_train = feats_train["y"].to_numpy(dtype=float)
        X_test = feats_test[self.base_columns].to_numpy(dtype=float)
        y_test = feats_test["y"].to_numpy(dtype=float)

        results = []

        for name, est in get_fast_estimators():
            try:
                model = clone(est)

                # Handle KNN special case
                if hasattr(model, 'n_neighbors'):
                    if isinstance(model, KNeighborsRegressor):
                        desired_k = getattr(model, "n_neighbors", 5)
                        max_allowed_k = max(1, min(desired_k, X_train.shape[0]))
                        if max_allowed_k != desired_k:
                            model.set_params(n_neighbors=max_allowed_k)

                # Time training
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time_s = float(time.time() - t0)

                # Time prediction
                t1 = time.time()
                preds = model.predict(X_test)
                predict_time_s = float(time.time() - t1)

                if len(preds) > 0:
                    se = (y_test - preds) ** 2
                    mse = np.nanmean(se)
                    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("inf")
                else:
                    rmse = float("inf")

                forecast_df = pd.DataFrame({
                    "ds": feats_test["ds"].to_numpy(),
                    "yhat": preds,
                })

                test_df = feats_test[["ds", "y"]].reset_index(drop=True)

                results.append({
                    "name": name,
                    "rmse": rmse,
                    "forecast_df": forecast_df,
                    "test_df": test_df,
                    "train_time_s": train_time_s,
                    "predict_time_s": predict_time_s,
                    "estimator_params": est.get_params(deep=True),
                })

            except Exception as e:
                results.append({
                    "name": name,
                    "rmse": float("inf"),
                    "forecast_df": pd.DataFrame(),
                    "test_df": pd.DataFrame(),
                    "train_time_s": None,
                    "predict_time_s": None,
                    "error": str(e),
                    "estimator_params": est.get_params(deep=True),
                })

        # Sort by RMSE
        results.sort(key=lambda r: r.get("rmse", float("inf")))
        return results

    def cross_validate_model(self, series_df: pd.DataFrame, estimator: BaseEstimator,
                           n_splits: int = 5) -> Dict[str, float]:
        """
        Cross-validation using time series splits to assess model stability.
        """
        ensure_daily_frequency(series_df)
        features_df, _ = self._prepare_features(series_df)

        train_feats = features_df.dropna(subset=self.base_columns + ["y"]).copy()
        X = train_feats[self.base_columns].to_numpy(dtype=float)
        y = train_feats["y"].to_numpy(dtype=float)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmse_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = clone(estimator)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_scores.append(rmse)

        return {
            "mean_rmse": np.mean(rmse_scores),
            "std_rmse": np.std(rmse_scores),
            "cv_scores": rmse_scores
        }


def get_fast_estimators() -> List[Tuple[str, object]]:
    return [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.01, max_iter=20000, tol=1e-4)),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, tol=1e-4)),
        ("BayesianRidge", BayesianRidge()),
        ("KNN", KNeighborsRegressor(n_neighbors=3)),
        ("ExtraTrees", ExtraTreesRegressor(random_state=42, n_estimators=200, n_jobs=-1)),
        ("AdaBoost", AdaBoostRegressor(random_state=42, n_estimators=200, learning_rate=0.05)),
        ("GB", GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3)),
        ("RF", RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=-1)),
        ("Huber", HuberRegressor(epsilon=1.35, max_iter=200, tol=1e-4)),
    ]


def _seasonal_naive_forecast(sorted_df: pd.DataFrame, period: int, test_start_ds: pd.Timestamp) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    """Seasonal naive: yhat_t = y_{t-period}."""
    test_df = sorted_df[sorted_df["ds"] >= test_start_ds].reset_index(drop=True)
    train_df = sorted_df[sorted_df["ds"] < test_start_ds].reset_index(drop=True)
    y_test = test_df["y"].to_numpy()
    # Map ds to index for quick lag lookups
    train_series = train_df.set_index("ds")["y"]
    preds = []
    for ds in test_df["ds"]:
        lag_ds = pd.Timestamp(ds) - pd.Timedelta(days=period)
        yhat = float(train_series.get(lag_ds, np.nan))
        preds.append(yhat)
    preds = np.array(preds, dtype=float)
    forecast_df = pd.DataFrame({"ds": test_df["ds"].to_numpy(), "yhat": preds})
    se = (y_test - preds) ** 2
    mse = np.nanmean(se)
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("inf")
    return rmse, forecast_df, test_df, 0.0, 0.0


 








def benchmark_models(series_df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    sorted_df = series_df.sort_values("ds").reset_index(drop=True)
    n = len(sorted_df)
    test_size = max(1, int(n * test_fraction))
    test_start_ds = sorted_df.iloc[-test_size]["ds"]

    # ML regressors
    for name, est in get_fast_estimators():
        try:
            rmse, forecast_df, test_df, train_time_s, predict_time_s = forecast_with_estimator(sorted_df, est, test_fraction)
            results.append({
                "name": name,
                "rmse": rmse,
                "forecast_df": forecast_df,
                "test_df": test_df,
                "train_time_s": train_time_s,
                "predict_time_s": predict_time_s,
                # Persist the exact hyperparameters so retrain uses the same configuration
                "estimator_params": est.get_params(deep=True),
            })
        except Exception as e:
            # Record the failure but keep the app running; this model will be ignored later
            results.append({
                "name": name,
                "rmse": float("inf"),
                "forecast_df": pd.DataFrame({"ds": [], "yhat": []}),
                "test_df": pd.DataFrame({"ds": [], "y": []}),
                "train_time_s": None,
                "predict_time_s": None,
                "error": str(e),
                "estimator_params": est.get_params(deep=True),
            })

    # Seasonal naive baseline (365) if enough history
    try:
        if n >= (365 + test_size + 1):
            rmse, forecast_df, test_df, tr_s, pr_s = _seasonal_naive_forecast(sorted_df, period=365, test_start_ds=test_start_ds)
            results.append({
                "name": "SeasonalNaive(365)",
                "rmse": rmse,
                "forecast_df": forecast_df,
                "test_df": test_df,
                "train_time_s": tr_s,
                "predict_time_s": pr_s,
                "estimator_params": {},
            })
    except Exception as e:
        results.append({
            "name": "SeasonalNaive(365)",
            "rmse": float("inf"),
            "forecast_df": pd.DataFrame({"ds": [], "yhat": []}),
            "test_df": pd.DataFrame({"ds": [], "y": []}),
            "train_time_s": None,
            "predict_time_s": None,
            "error": str(e),
            "estimator_params": {},
        })





    # Keep all models (including unavailable/failed baselines) so they appear in the UI table
    # Sort by RMSE with unavailable ones (inf) at the end
    results.sort(key=lambda r: r.get("rmse", float("inf")))
    return results


def train_full_and_forecast_future(series_df: pd.DataFrame, estimator, horizon_fraction: float = 0.2, horizon_steps: int = None) -> pd.DataFrame:
    if series_df.shape[0] < 10:
        raise ValueError("Insufficient data (need at least 10 rows)")
    ensure_daily_frequency(series_df)

    n = len(series_df)
    steps_to_forecast = max(1, int(n * horizon_fraction)) if horizon_steps is None else max(1, int(horizon_steps))

    # Prepare training features using the richer predictor set
    sorted_df = series_df.sort_values("ds").reset_index(drop=True)
    features_df, feature_cols_all = build_features(sorted_df)
    base_cols = _select_base_feature_columns(feature_cols_all)

    # Handle missing values more carefully for time series
    train_feats = features_df.dropna(subset=["y"]).copy()
    # Fill remaining NaN values in predictors with forward fill, then 0 for any remaining
    train_feats[base_cols] = train_feats[base_cols].fillna(method='ffill').fillna(0)

    if train_feats.empty:
        raise ValueError("No valid training rows after feature construction; cannot fit model for future forecast")

    model = clone(estimator)
    # Adapt K for KNN if the training set is very small
    try:
        if isinstance(model, KNeighborsRegressor):
            desired_k = getattr(model, "n_neighbors", 5)
            max_allowed_k = max(1, int(min(desired_k, train_feats.shape[0])))
            if max_allowed_k != desired_k:
                model.set_params(n_neighbors=max_allowed_k)
    except Exception:
        pass
    model.fit(train_feats[base_cols].to_numpy(dtype=float), train_feats["y"].to_numpy(dtype=float))

    # Seed history strictly with KNOWN y values to avoid NaNs in lag features
    hist = (
        sorted_df[["ds", "y"]]
        .dropna(subset=["y"])  # drop trailing future rows with missing target
        .reset_index(drop=True)
    )
    if hist.empty:
        raise ValueError("No known target values available to seed lags for forecasting")
    last_ds = pd.to_datetime(hist["ds"]).max()
    start_ds = pd.to_datetime(sorted_df["ds"]).min()

    def _dow_dummies_for(ds: pd.Timestamp, columns: list) -> Dict[str, int]:
        dow = int(pd.Timestamp(ds).dayofweek)
        present = {f"dow_{i}": 0 for i in range(7)}
        present[f"dow_{dow}"] = 1
        # Keep only columns expected by the model
        return {k: present.get(k, 0) for k in columns}

    dow_cols = [c for c in base_cols if c.startswith("dow_")]

    preds = []
    for step in range(1, steps_to_forecast + 1):
        next_ds = last_ds + pd.Timedelta(days=step)
        # Recreate engineered features for next_ds
        X_row = np.array([_build_row_for_date(base_cols, hist, next_ds, start_ds)])
        yhat = float(model.predict(X_row)[0])
        preds.append({"ds": next_ds, "yhat": yhat})
        # Append predicted value to history for next step's lags
        hist = pd.concat([hist, pd.DataFrame({"ds": [next_ds], "y": [yhat]})], ignore_index=True)

    return pd.DataFrame(preds)


def train_on_known_and_forecast_missing(full_df: pd.DataFrame, estimator, future_rows: pd.DataFrame) -> pd.DataFrame:
    if full_df.shape[0] < 10:
        raise ValueError("Insufficient data (need at least 10 rows)")
    ensure_daily_frequency(full_df)

    if "ds" not in future_rows.columns:
        raise ValueError("future_rows must include a 'ds' column")

    # Train on full history using the richer predictor set, but include exogenous
    sorted_df = full_df.sort_values("ds").reset_index(drop=True)
    features_df, feature_cols_all, info = build_features_internal(sorted_df, return_info=True)
    base_cols = _select_base_feature_columns(feature_cols_all)
    # Exclude exogenous columns from base when we explicitly append them via schema
    base_cols_no_exog = [c for c in base_cols if not (c.startswith("x_") or c.startswith("m_"))]
    # Include exogenous columns if present in training features
    exog_schema = info.get("exog_schema", [])
    exog_cols_expanded = []
    for sch in exog_schema:
        exog_cols_expanded.extend(list(sch.get("columns", [])))
    # Final training columns: non-exogenous time-series features + expanded exogenous features
    train_cols = base_cols_no_exog + exog_cols_expanded

    # Align retraining with benchmarking: require complete predictor rows
    # This avoids the distribution shift caused by imputing early lag NaNs.
    if train_cols:
        drop_subset = train_cols + ["y"]
    else:
        drop_subset = base_cols_no_exog + ["y"]
    train_feats = features_df.dropna(subset=drop_subset).copy()

    model = clone(estimator)
    # Adapt K for KNN if the training set is very small to avoid ValueError
    try:
        if isinstance(model, KNeighborsRegressor):
            desired_k = getattr(model, "n_neighbors", 5)
            max_allowed_k = max(1, int(min(desired_k, train_feats.shape[0])))
            if max_allowed_k != desired_k:
                model.set_params(n_neighbors=max_allowed_k)
    except Exception:
        pass
    X_train_mat = (
        train_feats[train_cols].to_numpy(dtype=float)
        if train_cols else train_feats[base_cols_no_exog].to_numpy(dtype=float)
    )
    y_train_vec = train_feats["y"].to_numpy(dtype=float)
    model.fit(X_train_mat, y_train_vec)

    # Prepare history strictly from KNOWN target rows to compute lags robustly
    hist = (
        sorted_df[["ds", "y"]]
        .dropna(subset=["y"])  # ignore any trailing rows with missing target
        .reset_index(drop=True)
    )
    if hist.empty:
        raise ValueError("No known target values available to seed lags for forecasting")

    start_ds = pd.to_datetime(sorted_df["ds"]).min()

    # Prepare a lookup for future exogenous values keyed by ds
    future_rows = future_rows.copy()
    future_rows["ds"] = pd.to_datetime(future_rows["ds"], errors="coerce")
    future_rows = future_rows.sort_values("ds").reset_index(drop=True)
    future_map = {}
    for _, rr in future_rows.iterrows():
        ds_val = rr.get("ds")
        if pd.isna(ds_val):
            continue
        # Keep original (non-engineered) columns for lookup
        row_dict = {k: rr[k] for k in full_df.columns if k not in ["ds", "y"] and k in future_rows.columns}
        future_map[pd.Timestamp(ds_val)] = row_dict

    # Helper to construct exogenous columns for a given ds based on training schema
    def _exog_vector_for_date(target_ds: pd.Timestamp) -> List[float]:
        values: List[float] = []
        raw = future_map.get(pd.Timestamp(target_ds), {})
        for sch in exog_schema:
            typ = sch.get("type")
            cols = list(sch.get("columns", []))
            if typ == "numeric":
                original_name = sch.get("original_name")
                new_col = cols[0] if cols else None
                val = pd.to_numeric(raw.get(original_name, pd.NA), errors="coerce")
                # Avoid introducing NaNs at prediction time; fall back to 0.0
                values.append(float(val) if not pd.isna(val) else 0.0)
            elif typ == "categorical":
                # One-hot across known training columns
                original_name = sch.get("original_name")
                raw_val = raw.get(original_name, pd.NA)
                raw_str = str(raw_val) if not pd.isna(raw_val) else "nan"
                # Build a map for quick hit
                hit_col = None
                for c in cols:
                    suffix = c.split(f"x_{sch.get('sanitized_base')}_", 1)[-1]
                    if raw_str == "nan" and suffix == "nan":
                        hit_col = c
                        break
                    if suffix == str(raw_val):
                        hit_col = c
                        break
                for c in cols:
                    values.append(1.0 if c == hit_col else 0.0)
            elif typ == "multilabel":
                original_name = sch.get("original_name")
                raw_val = raw.get(original_name, None)
                labels = []
                if isinstance(raw_val, str):
                    for delim in [",", ";", "|"]:
                        if delim in raw_val:
                            labels = [p.strip() for p in raw_val.split(delim) if p.strip()]
                            break
                    if not labels and raw_val.strip():
                        labels = [raw_val.strip()]
                # Map labels to columns; unknown labels are ignored
                label_set = set(labels)
                for c in cols:
                    # Column format: m_{base}_{label}
                    suffix = c.split(f"m_{sch.get('sanitized_base')}_", 1)[-1]
                    values.append(1.0 if suffix in label_set else 0.0)
            else:
                # Unknown exog type: align column count with schema and fill zeros
                for _ in cols:
                    values.append(0.0)
        return values

    preds = []
    last_processed_ds = hist["ds"].max()
    for _, r in future_rows.iterrows():
        current_ds = r["ds"]
        if pd.isna(current_ds):
            continue
        # If there is a gap larger than 1 day, we still simulate day-by-day to roll lags forward
        while (pd.Timestamp(current_ds) - pd.Timestamp(last_processed_ds)).days > 1:
            intermediate_ds = last_processed_ds + pd.Timedelta(days=1)
            base_vec = _build_row_for_date(base_cols_no_exog, hist, intermediate_ds, start_ds)
            exog_vec = _exog_vector_for_date(intermediate_ds) if exog_cols_expanded else []
            X_mid = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
            yhat_mid = float(model.predict(X_mid)[0])
            hist = pd.concat([hist, pd.DataFrame({"ds": [intermediate_ds], "y": [yhat_mid]})], ignore_index=True)
            last_processed_ds = intermediate_ds

        # Now predict for the requested date
        base_vec = _build_row_for_date(base_cols_no_exog, hist, current_ds, start_ds)
        exog_vec = _exog_vector_for_date(current_ds) if exog_cols_expanded else []
        X_row = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
        yhat = float(model.predict(X_row)[0])
        preds.append({"ds": current_ds, "yhat": yhat})
        hist = pd.concat([hist, pd.DataFrame({"ds": [current_ds], "y": [yhat]})], ignore_index=True)
        last_processed_ds = current_ds

    return pd.DataFrame(preds)


