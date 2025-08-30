from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
    BayesianRidge,
    TheilSenRegressor,
    RANSACRegressor,
    PassiveAggressiveRegressor,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, clone
import time
from sklearn.metrics import mean_absolute_percentage_error

from features import (
    build_features_internal,
    select_base_feature_columns,
    build_base_row_for_date,
    build_exog_vector_for_date,
)
from trend import fit_trend

# Import configuration
try:
    from config import DEFAULT_TEST_FRACTION, MIN_TRAINING_ROWS
except ImportError:
    # Fallback values
    DEFAULT_TEST_FRACTION = 0.2
    MIN_TRAINING_ROWS = 10
from typing import Callable

# -----------------------------
# Dynamic model registry
# -----------------------------
_MODEL_REGISTRY: Dict[str, Callable[[], BaseEstimator]] = {}

def register_model(name: str, factory: Callable[[], BaseEstimator]) -> None:
    """
    Register a model factory under a unique name.
    The factory will be called to create a fresh estimator instance when needed.
    """
    _MODEL_REGISTRY[name] = factory

def get_registered_estimators() -> List[Tuple[str, BaseEstimator]]:
    """Return a list of (name, estimator_instance) for all registered models."""
    estimators: List[Tuple[str, BaseEstimator]] = []
    for name, factory in _MODEL_REGISTRY.items():
        try:
            estimators.append((name, factory()))
        except Exception:
            # Skip factories that fail to construct
            continue
    return estimators

def _seed_default_models_if_empty() -> None:
    """Populate the registry with a default set of fast estimators if empty."""
    if _MODEL_REGISTRY:
        return
    # Linear models (reduced iter for speed)
    register_model("Ridge (alpha=0.5)", lambda: Ridge(alpha=0.5))
    register_model("Ridge (alpha=2.0)", lambda: Ridge(alpha=2.0))
    register_model("Lasso (alpha=0.1)", lambda: Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=10000, tol=1e-4))]))
    register_model("ElasticNet (alpha=0.01, l1=0.7)", lambda: Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=20000, tol=1e-4))]))
    register_model("BayesianRidge", lambda: BayesianRidge(alpha_1=1e-6, alpha_2=1e-6))
    # Neighbors
    register_model("KNN (n_neighbors=7)", lambda: KNeighborsRegressor(n_neighbors=7))
    # Trees / ensembles (trim heavy variants)
    register_model("AdaBoost (n_estimators=100)", lambda: AdaBoostRegressor(random_state=42, n_estimators=100, learning_rate=0.1))
    register_model("GB (n_estimators=100)", lambda: GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3))
    register_model("RF (n_estimators=200)", lambda: RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8, n_jobs=-1))
    # Robust linear
    register_model("Huber (epsilon=1.35)", lambda: Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, max_iter=5000, tol=1e-4))]))
    # Additional models for better coverage
    register_model("DecisionTree", lambda: DecisionTreeRegressor(random_state=42, max_depth=10))
    register_model("SVR (RBF)", lambda: Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel='rbf', C=1.0, gamma='scale'))]))
    register_model("SVR (Linear)", lambda: Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel='linear', C=1.0))]))
    register_model("LinearSVR", lambda: Pipeline([("scaler", StandardScaler()), ("model", LinearSVR(max_iter=5000, tol=1e-4))]))
    # Removed TheilSen and RANSACRegressor from default registry per user request
    register_model("PassiveAggressive", lambda: Pipeline([("scaler", StandardScaler()), ("model", PassiveAggressiveRegressor(random_state=42, max_iter=1000))]))
    register_model("SGD", lambda: Pipeline([("scaler", StandardScaler()), ("model", SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))]))
    # Removed MLPRegressor from default registry per user request

# Seed defaults on import so registry always has baseline models
_seed_default_models_if_empty()
 
# Removed Linear regression compatibility wrapper

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
            self.base_columns = select_base_feature_columns(feature_cols_all)
            self.exog_schema = info.get("exog_schema", [])

        if return_info:
            return features_df, feature_cols_all, info
        return features_df, feature_cols_all

    def benchmark_models(self, series_df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION) -> List[Dict[str, Any]]:
        """
        Unified benchmarking that uses the same feature engineering as retraining.
        This ensures consistency between model selection and retraining.
        """
        if series_df.shape[0] < MIN_TRAINING_ROWS:
            raise ValueError("Insufficient data (need at least 10 rows)")

        # Prepare sorted data and split boundary
        sorted_df = series_df.sort_values("ds").reset_index(drop=True)
        n = len(sorted_df)
        test_size = max(1, int(n * test_fraction))
        test_start_ds = sorted_df.iloc[-test_size]["ds"]
        train_df = sorted_df[sorted_df["ds"] < test_start_ds].reset_index(drop=True)
        test_df_raw = sorted_df[sorted_df["ds"] >= test_start_ds].reset_index(drop=True)

        # Build features ONLY on the training slice to avoid leakage
        features_train, feature_cols_all, info = build_features_internal(train_df, return_info=True)
        # Initialize trainer state from train slice
        if not self.feature_columns:
            self.feature_columns = feature_cols_all
            self.base_columns = select_base_feature_columns(feature_cols_all)
            self.exog_schema = info.get("exog_schema", [])

        # Drop rows with missing predictors/target on train only
        feats_train = features_train.dropna(subset=self.base_columns + ["y"]).copy()

        if feats_train.empty or test_df_raw.empty:
            return [{"name": "NoModels", "rmse": float("inf"), "forecast_df": pd.DataFrame(), "test_df": pd.DataFrame()}]

        # Prepare training arrays (raw target)
        X_train = feats_train[self.base_columns].to_numpy(dtype=float)
        y_train_raw = feats_train["y"].to_numpy(dtype=float)

        # Fit trend on both training slice and full known series; use the stronger one
        try:
            trend_model_train = fit_trend(train_df)
            try:
                known_full = series_df.dropna(subset=["y"]).copy()
            except Exception:
                known_full = train_df
            trend_model_full = fit_trend(known_full)
            deg_train = int(getattr(trend_model_train, "degree", 0))
            deg_full = int(getattr(trend_model_full, "degree", 0))
            trend_model = trend_model_train if deg_train >= deg_full else trend_model_full
            trend_fit_train = trend_model.fitted(feats_train["ds"]) if "ds" in feats_train.columns else pd.Series(np.zeros_like(y_train_raw))
            y_train_det = (y_train_raw - trend_fit_train.to_numpy(dtype=float))
            has_meaningful_trend = int(getattr(trend_model, "degree", 0)) > 0
        except Exception:
            trend_model = None
            y_train_det = y_train_raw.copy()
            has_meaningful_trend = False

        results = []

        # Use dynamic registry (seed defaults on first use)
        _seed_default_models_if_empty()
        for name, est in get_fast_estimators():
            try:
                # Train RAW variant
                model_raw = clone(est)

                # Handle KNN special case
                if hasattr(model_raw, 'n_neighbors'):
                    if isinstance(model_raw, KNeighborsRegressor):
                        desired_k = getattr(model_raw, "n_neighbors", 5)
                        max_allowed_k = max(1, min(desired_k, X_train.shape[0]))
                        if max_allowed_k != desired_k:
                            model_raw.set_params(n_neighbors=max_allowed_k)

                # Time training
                t0 = time.time()
                model_raw.fit(X_train, y_train_raw)
                train_time_raw = float(time.time() - t0)

                # Walk-forward prediction on the test slice WITHOUT using its targets or engineered features
                t1 = time.time()
                # History seeded strictly from training known targets
                hist = train_df[["ds", "y"]].dropna(subset=["y"]).reset_index(drop=True)
                if hist.empty:
                    raise ValueError("No history available for evaluation")
                start_ds = pd.to_datetime(train_df["ds"]).min()

                # Prepare exogenous lookup from raw test rows
                future_map = {}
                for _, rr in test_df_raw.iterrows():
                    ds_val = pd.to_datetime(rr.get("ds"), errors="coerce")
                    if pd.isna(ds_val):
                        continue
                    row_dict = {k: rr[k] for k in sorted_df.columns if k not in ["ds", "y"] and k in test_df_raw.columns}
                    future_map[pd.Timestamp(ds_val)] = row_dict

                # Separate base cols into non-exogenous and use schema for exogenous
                base_cols_no_exog = [c for c in self.base_columns if not (c.startswith("x_") or c.startswith("m_"))]

                def _walk_forward(model_obj: Any, add_trend: bool) -> Dict[str, Any]:
                    preds_records = []
                    y_test_vals = []
                    test_ds_vals = []
                    local_hist = hist.copy()
                    last_processed_ds = local_hist["ds"].max()
                    for _, r in test_df_raw.iterrows():
                        current_ds = pd.to_datetime(r["ds"], errors="coerce")
                        if pd.isna(current_ds):
                            continue
                        while (pd.Timestamp(current_ds) - pd.Timestamp(last_processed_ds)).days > 1:
                            intermediate_ds = last_processed_ds + pd.Timedelta(days=1)
                            base_vec = build_base_row_for_date(base_cols_no_exog, local_hist, intermediate_ds, start_ds)
                            exog_vec = build_exog_vector_for_date(intermediate_ds, future_map, self.exog_schema)
                            X_mid = np.array([base_vec + exog_vec]) if len(exog_vec) > 0 else np.array([base_vec])
                            yhat_mid = float(model_obj.predict(X_mid)[0])
                            if add_trend and trend_model is not None:
                                try:
                                    yhat_mid += float(trend_model.extrapolate(pd.Series([intermediate_ds])).iloc[0])
                                except Exception:
                                    pass
                            local_hist = pd.concat([local_hist, pd.DataFrame({"ds": [intermediate_ds], "y": [yhat_mid]})], ignore_index=True)
                            last_processed_ds = intermediate_ds

                        base_vec = build_base_row_for_date(base_cols_no_exog, local_hist, current_ds, start_ds)
                        exog_vec = build_exog_vector_for_date(current_ds, future_map, self.exog_schema)
                        X_row = np.array([base_vec + exog_vec]) if len(exog_vec) > 0 else np.array([base_vec])
                        yhat = float(model_obj.predict(X_row)[0])
                        if add_trend and trend_model is not None:
                            try:
                                yhat += float(trend_model.extrapolate(pd.Series([current_ds])).iloc[0])
                            except Exception:
                                pass
                        preds_records.append(yhat)
                        y_test_vals.append(float(pd.to_numeric(r.get("y"), errors="coerce")))
                        test_ds_vals.append(pd.Timestamp(current_ds))
                        local_hist = pd.concat([local_hist, pd.DataFrame({"ds": [current_ds], "y": [yhat]})], ignore_index=True)
                        last_processed_ds = current_ds

                    preds_arr = np.array(preds_records, dtype=float)
                    y_test_arr = np.array(y_test_vals, dtype=float)
                    if preds_arr.size > 0 and y_test_arr.size == preds_arr.size:
                        se = (y_test_arr - preds_arr) ** 2
                        mse = np.nanmean(se)
                        rmse_val = float(np.sqrt(mse)) if np.isfinite(mse) else float("inf")
                        try:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                abs_err = np.abs(y_test_arr - preds_arr)
                                denom = np.abs(y_test_arr)
                                mask = denom > 1e-12
                                ratios = np.zeros_like(abs_err, dtype=float)
                                ratios[mask] = abs_err[mask] / denom[mask]
                                mape_val = float(np.nanmean(ratios)) if np.any(mask) else None
                        except Exception:
                            mape_val = None
                        try:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                denom2 = (np.abs(y_test_arr) + np.abs(preds_arr))
                                mask2 = denom2 > 1e-12
                                smape_vals = np.zeros_like(y_test_arr, dtype=float)
                                smape_vals[mask2] = (2.0 * np.abs(preds_arr[mask2] - y_test_arr[mask2])) / denom2[mask2]
                                smape_val = float(np.nanmean(smape_vals)) if np.any(mask2) else None
                        except Exception:
                            smape_val = None
                    else:
                        rmse_val = float("inf")
                        mape_val = None
                        smape_val = None

                    forecast_df = pd.DataFrame({"ds": test_ds_vals, "yhat": preds_arr})
                    test_df = pd.DataFrame({"ds": test_ds_vals, "y": y_test_arr})
                    return {"forecast_df": forecast_df, "test_df": test_df, "rmse": rmse_val, "mape": mape_val, "smape": smape_val}

                # RAW evaluation
                res_raw = _walk_forward(model_raw, add_trend=False)
                predict_time_raw = float(time.time() - t1)

                # Keep full unique registry name so parameterized variants are independent
                results.append({
                    "name": name,
                    "rmse": res_raw["rmse"],
                    "mape": res_raw["mape"],
                    "smape": res_raw["smape"],
                    "forecast_df": res_raw["forecast_df"],
                    "test_df": res_raw["test_df"],
                    "train_time_s": train_time_raw,
                    "predict_time_s": predict_time_raw,
                    "estimator_params": est.get_params(deep=True),
                })

                # DETRENDED variant - only if meaningful trend detected
                if has_meaningful_trend:
                    model_det = clone(est)
                    # KNN adjustment again (X same)
                    if hasattr(model_det, 'n_neighbors'):
                        if isinstance(model_det, KNeighborsRegressor):
                            desired_k = getattr(model_det, "n_neighbors", 5)
                            max_allowed_k = max(1, min(desired_k, X_train.shape[0]))
                            if max_allowed_k != desired_k:
                                model_det.set_params(n_neighbors=max_allowed_k)
                    t2 = time.time()
                    model_det.fit(X_train, y_train_det)
                    train_time_det = float(time.time() - t2)

                    t3 = time.time()
                    res_det = _walk_forward(model_det, add_trend=True)
                    predict_time_det = float(time.time() - t3)

                    results.append({
                        "name": f"{name}+Trend",
                        "rmse": res_det["rmse"],
                        "mape": res_det["mape"],
                        "smape": res_det["smape"],
                        "forecast_df": res_det["forecast_df"],
                        "test_df": res_det["test_df"],
                        "train_time_s": train_time_det,
                        "predict_time_s": predict_time_det,
                        "estimator_params": est.get_params(deep=True),
                    })

            except Exception as e:
                results.append({
                    "name": name,
                    "rmse": float("inf"),
                    "mape": None,
                    "smape": None,
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


def get_fast_estimators() -> List[Tuple[str, BaseEstimator]]:
    """
    Backwards-compatible accessor used throughout the app. Now returns the
    dynamically registered estimators. If no models are registered yet,
    default fast estimators are seeded.
    """
    _seed_default_models_if_empty()
    return get_registered_estimators()


def train_on_known_and_forecast_missing(full_df: pd.DataFrame, estimator, future_rows: pd.DataFrame) -> pd.DataFrame:
    if full_df.shape[0] < 10:
        raise ValueError("Insufficient data (need at least 10 rows)")

    if "ds" not in future_rows.columns:
        raise ValueError("future_rows must include a 'ds' column")

    # Train using the SAME pipeline/settings as benchmarking, but on ALL known rows (no leakage)
    # Build features/schema strictly from known-target history (identical approach, different data size)
    sorted_df = full_df.sort_values("ds").reset_index(drop=True)
    known_df = sorted_df.dropna(subset=["y"]).reset_index(drop=True)
    features_df, feature_cols_all, info = build_features_internal(known_df, return_info=True)
    base_cols = select_base_feature_columns(feature_cols_all)
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

    # Prepare history strictly from KNOWN target rows to compute lags robustly (no leakage)
    hist = (
        known_df[["ds", "y"]]
        .reset_index(drop=True)
    )
    if hist.empty:
        raise ValueError("No known target values available to seed lags for forecasting")

    # Align trend origin with training (known-only) just like benchmarking
    start_ds = pd.to_datetime(known_df["ds"]).min()

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
            base_vec = build_base_row_for_date(base_cols_no_exog, hist, intermediate_ds, start_ds)
            exog_vec = _exog_vector_for_date(intermediate_ds) if exog_cols_expanded else []
            X_mid = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
            yhat_mid = float(model.predict(X_mid)[0])
            hist = pd.concat([hist, pd.DataFrame({"ds": [intermediate_ds], "y": [yhat_mid]})], ignore_index=True)
            last_processed_ds = intermediate_ds

        # Now predict for the requested date
        base_vec = build_base_row_for_date(base_cols_no_exog, hist, current_ds, start_ds)
        exog_vec = _exog_vector_for_date(current_ds) if exog_cols_expanded else []
        X_row = np.array([base_vec + exog_vec]) if exog_cols_expanded else np.array([base_vec])
        yhat = float(model.predict(X_row)[0])
        preds.append({"ds": current_ds, "yhat": yhat})
        hist = pd.concat([hist, pd.DataFrame({"ds": [current_ds], "y": [yhat]})], ignore_index=True)
        last_processed_ds = current_ds

    return pd.DataFrame(preds)


