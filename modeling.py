from typing import Tuple, List, Dict

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.base import clone
import time

from features import build_features, build_features_internal

try:
    # Optional import for SARIMA baseline
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # keep optional
    SARIMAX = None

# Optional Prophet baseline
try:
    from prophet import Prophet
except Exception:
    Prophet = None

def _select_base_feature_columns(all_columns: List[str]) -> List[str]:
    """Pick model features we can also recreate when forecasting.

    Includes: lags, moving averages, DOW dummies, trend, and Fourier terms.
    """
    prefixes = ("lag", "ma", "dow_", "fourier_")
    base_cols = [c for c in all_columns if c.startswith(prefixes) or c == "trend"]
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


def forecast_with_estimator(series_df: pd.DataFrame, estimator, test_fraction: float = 0.2) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    if series_df.shape[0] < 10:
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

    # Drop rows with missing predictors/target
    feats_train = feats_train.dropna(subset=base_cols + ["y"]) if not feats_train.empty else feats_train
    feats_test = feats_test.dropna(subset=base_cols + ["y"]) if not feats_test.empty else feats_test

    # Guard against empty splits after dropna. If no test rows remain, return inf score to exclude this model.
    if feats_test.empty or ("y" not in feats_test.columns):
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0

    # If no train rows remain, propagate a clear error to be reported by the caller
    if feats_train.empty or ("y" not in feats_train.columns):
        raise ValueError("No valid training rows after feature construction; cannot fit model")

    X_train = feats_train[base_cols].to_numpy()
    y_train = feats_train["y"].to_numpy()
    X_test = feats_test[base_cols].to_numpy()
    y_test = feats_test["y"].to_numpy()

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
        denom = np.where(np.abs(y_test) > 1e-8, np.abs(y_test), np.nan)
        ape = np.abs((y_test - preds) / denom)
        _val = np.nanmean(ape)
        mape = float(_val) if np.isfinite(_val) else float("inf")
    else:
        mape = float("inf")

    # Convert feats_test back to original structure for the app (only ds/y)
    test_df = feats_test[["ds", "y"]].reset_index(drop=True)

    return mape, forecast_df, test_df, train_time_s, predict_time_s


def get_fast_estimators() -> List[Tuple[str, object]]:
    return [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.001, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)),
        ("BayesianRidge", BayesianRidge()),
        ("KNN", KNeighborsRegressor(n_neighbors=3)),
        ("DecisionTree", DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=3)),
        ("ExtraTrees", ExtraTreesRegressor(random_state=42, n_estimators=200, n_jobs=-1)),
        ("AdaBoost", AdaBoostRegressor(random_state=42, n_estimators=200, learning_rate=0.05)),
        ("GB", GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3)),
        ("RF", RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=-1)),
        ("SVR", SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="scale")),
        ("Huber", HuberRegressor()),
        ("MLP", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1500, random_state=42)),
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
    denom = np.where(np.abs(y_test) > 1e-8, np.abs(y_test), np.nan)
    ape = np.abs((y_test - preds) / denom)
    mape = float(np.nanmean(ape))
    return mape, forecast_df, test_df, 0.0, 0.0


 


def _sarima_forecast(sorted_df: pd.DataFrame, test_start_ds: pd.Timestamp) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    if SARIMAX is None:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    train = sorted_df[sorted_df["ds"] < test_start_ds].copy()
    test = sorted_df[sorted_df["ds"] >= test_start_ds].copy()
    if train.empty or test.empty:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    # Coerce and clean
    train["y"] = pd.to_numeric(train["y"], errors="coerce")
    train = train.dropna(subset=["y"])  # SARIMAX cannot handle NaNs in endog
    if train.empty:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    y_train = train["y"].astype(float).to_numpy()
    # Dynamic seasonal period selection for robustness on short histories
    len_train = len(train)
    seasonal_period = 365 if len_train >= 730 else (7 if len_train >= 14 else None)
    t0 = time.time()
    if seasonal_period is None:
        model = SARIMAX(
            y_train,
            order=(8, 1, 0),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        model = SARIMAX(
            y_train,
            order=(8, 1, 0),
            seasonal_order=(1, 1, 0, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    fit = model.fit(disp=False)
    train_time_s = float(time.time() - t0)
    t1 = time.time()
    preds = fit.forecast(steps=len(test))
    predict_time_s = float(time.time() - t1)
    preds = preds.astype(float)
    forecast_df = pd.DataFrame({"ds": test["ds"].to_numpy(), "yhat": preds})
    denom = np.where(np.abs(test["y"].to_numpy()) > 1e-8, np.abs(test["y"].to_numpy()), np.nan)
    ape = np.abs((test["y"].to_numpy() - preds) / denom)
    mape = float(np.nanmean(ape))
    return mape, forecast_df, test.reset_index(drop=True), train_time_s, predict_time_s


# Prophet baseline (optional)
def _prophet_forecast(sorted_df: pd.DataFrame, test_start_ds: pd.Timestamp) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    if Prophet is None:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    train = sorted_df[sorted_df["ds"] < test_start_ds].copy()
    test = sorted_df[sorted_df["ds"] >= test_start_ds].copy()
    if train.empty or test.empty:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    train = train[["ds", "y"]].copy()
    # Coerce types for Prophet robustness
    train["ds"] = pd.to_datetime(train["ds"], errors="coerce")
    train["y"] = pd.to_numeric(train["y"], errors="coerce")
    train = train.dropna()
    if train.empty:
        return float("inf"), pd.DataFrame({"ds": [], "yhat": []}), pd.DataFrame({"ds": [], "y": []}), 0.0, 0.0
    t0 = time.time()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(train.rename(columns={"ds": "ds", "y": "y"}))
    train_time_s = float(time.time() - t0)
    t1 = time.time()
    future = pd.DataFrame({"ds": pd.to_datetime(test["ds"].to_numpy())})
    forecast = model.predict(future)
    predict_time_s = float(time.time() - t1)
    preds = forecast["yhat"].astype(float).to_numpy()
    forecast_df = pd.DataFrame({"ds": test["ds"].to_numpy(), "yhat": preds})
    denom = np.where(np.abs(test["y"].to_numpy()) > 1e-8, np.abs(test["y"].to_numpy()), np.nan)
    ape = np.abs((test["y"].to_numpy() - preds) / denom)
    mape = float(np.nanmean(ape))
    return mape, forecast_df, test.reset_index(drop=True), train_time_s, predict_time_s


def benchmark_models(series_df: pd.DataFrame, test_fraction: float = 0.2) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    sorted_df = series_df.sort_values("ds").reset_index(drop=True)
    n = len(sorted_df)
    test_size = max(1, int(n * test_fraction))
    test_start_ds = sorted_df.iloc[-test_size]["ds"]

    # ML regressors
    for name, est in get_fast_estimators():
        try:
            mape, forecast_df, test_df, train_time_s, predict_time_s = forecast_with_estimator(sorted_df, est, test_fraction)
            results.append({
                "name": name,
                "mape": mape,
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
                "mape": float("inf"),
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
            mape, forecast_df, test_df, tr_s, pr_s = _seasonal_naive_forecast(sorted_df, period=365, test_start_ds=test_start_ds)
            results.append({
                "name": "SeasonalNaive(365)",
                "mape": mape,
                "forecast_df": forecast_df,
                "test_df": test_df,
                "train_time_s": tr_s,
                "predict_time_s": pr_s,
                "estimator_params": {},
            })
    except Exception as e:
        results.append({
            "name": "SeasonalNaive(365)",
            "mape": float("inf"),
            "forecast_df": pd.DataFrame({"ds": [], "yhat": []}),
            "test_df": pd.DataFrame({"ds": [], "y": []}),
            "train_time_s": None,
            "predict_time_s": None,
            "error": str(e),
            "estimator_params": {},
        })

    # SARIMA baseline (may be slow)
    try:
        mape, forecast_df, test_df, tr_s, pr_s = _sarima_forecast(sorted_df, test_start_ds)
        results.append({
            "name": "SARIMA(8,1,0)x(1,1,0,365)",
            "mape": mape,
            "forecast_df": forecast_df,
            "test_df": test_df,
            "train_time_s": tr_s,
            "predict_time_s": pr_s,
            "estimator_params": {},
        })
    except Exception as e:
        results.append({
            "name": "SARIMA(8,1,0)x(1,1,0,365)",
            "mape": float("inf"),
            "forecast_df": pd.DataFrame({"ds": [], "yhat": []}),
            "test_df": pd.DataFrame({"ds": [], "y": []}),
            "train_time_s": None,
            "predict_time_s": None,
            "error": str(e),
            "estimator_params": {},
        })

    # Prophet baseline (optional)
    try:
        mape, forecast_df, test_df, tr_s, pr_s = _prophet_forecast(sorted_df, test_start_ds)
        results.append({
            "name": "Prophet(Y+W)",
            "mape": mape,
            "forecast_df": forecast_df,
            "test_df": test_df,
            "train_time_s": tr_s,
            "predict_time_s": pr_s,
            "estimator_params": {},
        })
    except Exception as e:
        results.append({
            "name": "Prophet(Y+W)",
            "mape": float("inf"),
            "forecast_df": pd.DataFrame({"ds": [], "yhat": []}),
            "test_df": pd.DataFrame({"ds": [], "y": []}),
            "train_time_s": None,
            "predict_time_s": None,
            "error": str(e),
            "estimator_params": {},
        })

    # Keep all models (including unavailable/failed baselines) so they appear in the UI table
    # Sort by MAPE with unavailable ones (inf) at the end
    results.sort(key=lambda r: r.get("mape", float("inf")))
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
    train_feats = features_df.dropna(subset=base_cols + ["y"]).copy()

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
    model.fit(train_feats[base_cols].to_numpy(), train_feats["y"].to_numpy())

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
    # Include exogenous columns if present in training features
    exog_schema = info.get("exog_schema", [])
    exog_cols_expanded = []
    for sch in exog_schema:
        exog_cols_expanded.extend(list(sch.get("columns", [])))
    train_cols = base_cols + exog_cols_expanded

    train_feats = features_df.dropna(subset=train_cols + ["y"]).copy() if train_cols else features_df.dropna(subset=base_cols + ["y"]).copy()

    model = clone(estimator)
    model.fit(train_feats[train_cols].to_numpy() if train_cols else train_feats[base_cols].to_numpy(), train_feats["y"].to_numpy())

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
                if pd.isna(val):
                    # Keep NaN; model will error if NaN was not seen during train; our checklist should prevent this
                    values.append(np.nan)
                else:
                    values.append(float(val))
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
            base_vec = _build_row_for_date(base_cols, hist, intermediate_ds, start_ds)
            exog_vec = _exog_vector_for_date(intermediate_ds) if exog_cols_expanded else []
            X_mid = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
            yhat_mid = float(model.predict(X_mid)[0])
            hist = pd.concat([hist, pd.DataFrame({"ds": [intermediate_ds], "y": [yhat_mid]})], ignore_index=True)
            last_processed_ds = intermediate_ds

        # Now predict for the requested date
        base_vec = _build_row_for_date(base_cols, hist, current_ds, start_ds)
        exog_vec = _exog_vector_for_date(current_ds) if exog_cols_expanded else []
        X_row = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
        yhat = float(model.predict(X_row)[0])
        preds.append({"ds": current_ds, "yhat": yhat})
        hist = pd.concat([hist, pd.DataFrame({"ds": [current_ds], "y": [yhat]})], ignore_index=True)
        last_processed_ds = current_ds

    return pd.DataFrame(preds)


