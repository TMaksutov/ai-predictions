from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.preprocessing import MultiLabelBinarizer
import time


def _ensure_daily_frequency(series_df: pd.DataFrame) -> None:
    """
    Ensure the series has a daily frequency with no gaps.
    Raises ValueError otherwise.
    """
    if series_df.empty:
        raise ValueError("Empty series provided")

    # Sort and check inferred frequency
    series_df = series_df.sort_values("ds").reset_index(drop=True)
    inferred = pd.infer_freq(series_df["ds"])
    if inferred != "D":
        raise ValueError(
            "Dataset must be daily frequency ('D') with regular 1-day intervals. "
            "Please resample/fill gaps before uploading."
        )


def _sanitize_feature_token(token: str) -> str:
    """Sanitize category/label tokens for safe column names."""
    if token is None:
        return "unknown"
    s = str(token)
    # Replace spaces and disallowed chars with underscores and collapse repeats
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    sanitized = "".join(out)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "unknown"


def _parse_multilabel_cell(value) -> List[str]:
    """Parse a potential multi-label cell into list of labels."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value)
    # Split on common delimiters
    for delim in [",", ";", "|"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            return [p for p in parts if p]
    # No delimiter, treat as single label if not empty and not 'nan'
    s = s.strip()
    if s and s.lower() != "nan":
        return [s]
    return []


def _is_probably_multilabel(series: pd.Series) -> bool:
    """Heuristic: if any row contains a list-like of 2+ labels after parsing."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    for val in sample:
        labels = _parse_multilabel_cell(val)
        if len(labels) >= 2:
            return True
    return False


def _build_features_internal(series_df: pd.DataFrame, return_info: bool = False) -> Tuple[pd.DataFrame, list, dict]:
    """
    Build regression features:
    - Lags: 1 and 7 days (impute missing with median of y instead of dropping rows)
    - Day-of-week one-hot (full one-hot; no drop_first)

    Returns (features_df, feature_columns, info_dict)
    info_dict keys:
      - lag_imputed_rows: int
      - num_dow_dummies: int
      - total_feature_columns: int
    """
    df = series_df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    # Construct lag features
    df["lag1"] = df["y"].shift(1)
    df["lag7"] = df["y"].shift(7)
    original_lag_na = ((df["lag1"].isna()) | (df["lag7"].isna())).sum()

    # Impute lag NaNs with median of y to avoid dropping rows
    y_median = float(df["y"].median()) if not df["y"].dropna().empty else 0.0
    df["lag1"] = df["lag1"].fillna(y_median)
    df["lag7"] = df["lag7"].fillna(y_median)

    # Day-of-week as full one-hot
    df["dow"] = df["ds"].dt.dayofweek.astype(int)
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=False)

    features = pd.concat([df[["ds", "y", "lag1", "lag7"]], dow_dummies], axis=1)

    feature_cols = ["lag1", "lag7"] + list(dow_dummies.columns)

    # Add exogenous features if present
    exog_cols = [c for c in df.columns if c not in ["ds", "y", "lag1", "lag7", "dow"]]
    num_exog_numeric = 0
    num_exog_categorical = 0
    num_exog_multilabel = 0

    for col in exog_cols:
        series = df[col]
        # Detect numeric-like
        coerced = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(coerced.notna().sum()) / float(len(series)) if len(series) > 0 else 0.0
        if numeric_ratio >= 0.6:
            # Treat as numeric exogenous; impute via ffill/bfill then median
            ser_num = coerced.fillna(method="ffill").fillna(method="bfill")
            median_val = float(ser_num.median()) if not np.isnan(ser_num.median()) else 0.0
            ser_num = ser_num.fillna(median_val)
            new_name = f"exog_{_sanitize_feature_token(col)}"
            features[new_name] = ser_num.astype(float)
            feature_cols.append(new_name)
            num_exog_numeric += 1
            continue

        # Non-numeric: decide multi-label vs single-label
        if _is_probably_multilabel(series):
            # Multi-label binarization
            parsed = series.apply(_parse_multilabel_cell)
            mlb = MultiLabelBinarizer()
            binarized = mlb.fit_transform(parsed)
            # Build DataFrame with sanitized class names
            binarized_cols = [f"ml_{_sanitize_feature_token(col)}__{_sanitize_feature_token(c)}" for c in mlb.classes_]
            if binarized.shape[1] == 0:
                # No classes; skip
                continue
            bin_df = pd.DataFrame(binarized, columns=binarized_cols, index=features.index)
            features = pd.concat([features, bin_df], axis=1)
            feature_cols.extend(binarized_cols)
            num_exog_multilabel += 1
        else:
            # Single-label categorical one-hot
            ser_cat = series.fillna("Unknown").astype(str)
            dummies = pd.get_dummies(ser_cat, prefix=f"exog_{_sanitize_feature_token(col)}", drop_first=False)
            if not dummies.empty:
                features = pd.concat([features, dummies], axis=1)
                feature_cols.extend(list(dummies.columns))
                num_exog_categorical += 1

    info = {
        "lag_imputed_rows": int(original_lag_na),
        "num_dow_dummies": int(len(dow_dummies.columns)),
        "total_feature_columns": int(len(feature_cols)),
        "num_exog_numeric": int(num_exog_numeric),
        "num_exog_categorical": int(num_exog_categorical),
        "num_exog_multilabel": int(num_exog_multilabel),
    }

    if return_info:
        return features.reset_index(drop=True), feature_cols, info
    return features.reset_index(drop=True), feature_cols, {}


def _build_features(series_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    features, feature_cols, _ = _build_features_internal(series_df, return_info=False)
    return features, feature_cols


def forecast_regression(
    series_df: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Forecast using a simple multiple linear regression with lag and calendar features.

    Splits the series into train/test by time, fits on train, predicts the last
    test_fraction window. Returns (nrmse, forecast_df, test_df), where nrmse is
    RMSE normalized by the range of y_true on the test set.
    """
    if series_df.shape[0] < 30:
        raise ValueError("Insufficient data (need at least 30 rows)")

    # Ensure daily frequency
    _ensure_daily_frequency(series_df)

    n = len(series_df)
    test_size = max(1, int(n * test_fraction))
    test_start_ds = series_df.sort_values("ds").iloc[-test_size]["ds"]

    # Build features for the whole series then split by date
    features, feature_cols = _build_features(series_df)

    train_mask = features["ds"] < test_start_ds
    test_mask = features["ds"] >= test_start_ds

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Not enough rows after feature construction for train/test split")

    X_train = features.loc[train_mask, feature_cols].to_numpy()
    y_train = features.loc[train_mask, "y"].to_numpy()

    X_test = features.loc[test_mask, feature_cols].to_numpy()
    y_test = features.loc[test_mask, "y"].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    # Build forecast dataframe aligned to test window
    forecast_df = pd.DataFrame({
        "ds": features.loc[test_mask, "ds"].to_numpy(),
        "yhat": yhat,
    })

    # Compute nRMSE over the aligned test rows
    rmse = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
    y_range = float(np.max(y_test) - np.min(y_test))
    nrmse = rmse / y_range if y_range > 0 else rmse

    # Original test frame from the raw series (for plotting split line)
    test_df = series_df[series_df["ds"] >= test_start_ds].reset_index(drop=True)

    return nrmse, forecast_df, test_df


def forecast_with_estimator(
    series_df: pd.DataFrame,
    estimator,
    test_fraction: float = 0.2,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    """
    Generic helper to forecast with any sklearn regressor that implements fit/predict.
    Returns (nrmse, forecast_df, test_df, train_time_s, predict_time_s).
    """
    if series_df.shape[0] < 30:
        raise ValueError("Insufficient data (need at least 30 rows)")

    _ensure_daily_frequency(series_df)

    n = len(series_df)
    test_size = max(1, int(n * test_fraction))
    test_start_ds = series_df.sort_values("ds").iloc[-test_size]["ds"]

    features, feature_cols = _build_features(series_df)

    train_mask = features["ds"] < test_start_ds
    test_mask = features["ds"] >= test_start_ds

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Not enough rows after feature construction for train/test split")

    X_train = features.loc[train_mask, feature_cols].to_numpy()
    y_train = features.loc[train_mask, "y"].to_numpy()

    X_test = features.loc[test_mask, feature_cols].to_numpy()
    y_test = features.loc[test_mask, "y"].to_numpy()

    model = clone(estimator)

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = float(time.perf_counter() - train_start)

    predict_start = time.perf_counter()
    yhat = model.predict(X_test)
    predict_time_s = float(time.perf_counter() - predict_start)

    forecast_df = pd.DataFrame({
        "ds": features.loc[test_mask, "ds"].to_numpy(),
        "yhat": yhat,
    })

    rmse = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
    y_range = float(np.max(y_test) - np.min(y_test))
    nrmse = rmse / y_range if y_range > 0 else rmse

    test_df = series_df[series_df["ds"] >= test_start_ds].reset_index(drop=True)

    return nrmse, forecast_df, test_df, train_time_s, predict_time_s


def get_fast_estimators() -> List[Tuple[str, object]]:
    """
    Returns a list of (name, estimator) pairs of relatively fast sklearn regressors.
    """
    return [
        ("Linear", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.001, max_iter=10000)),
        ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("GB", GradientBoostingRegressor(random_state=42, n_estimators=100)),
        ("Huber", HuberRegressor()),
    ]


def benchmark_models(
    series_df: pd.DataFrame, test_fraction: float = 0.2
) -> List[Dict[str, object]]:
    """
    Fit several fast sklearn models on the same train/test split and return
    a list of results sorted by nRMSE ascending. Each item contains:
      { 'name': str, 'nrmse': float, 'forecast_df': pd.DataFrame, 'test_df': pd.DataFrame }
    """
    results: List[Dict[str, object]] = []
    for name, est in get_fast_estimators():
        nrmse, forecast_df, test_df, train_time_s, predict_time_s = forecast_with_estimator(series_df, est, test_fraction)
        results.append({
            "name": name,
            "nrmse": nrmse,
            "forecast_df": forecast_df,
            "test_df": test_df,
            "train_time_s": train_time_s,
            "predict_time_s": predict_time_s,
        })

    results.sort(key=lambda r: r["nrmse"])
    return results



def train_full_and_forecast_future(
    series_df: pd.DataFrame,
    estimator,
    horizon_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Train the provided estimator on the full available history (using the same
    lag/calendar features) and iteratively forecast forward for
    int(len(series_df) * horizon_fraction) daily steps.

    Returns a DataFrame with columns ['ds', 'yhat'] for the future horizon.
    """
    if series_df.shape[0] < 30:
        raise ValueError("Insufficient data (need at least 30 rows)")

    _ensure_daily_frequency(series_df)

    # Build features over the full history and fit on all available rows
    features, feature_cols = _build_features(series_df)
    if features.empty:
        raise ValueError("Not enough rows after feature construction to train")

    X_full = features[feature_cols].to_numpy()
    y_full = features["y"].to_numpy()

    model = clone(estimator)
    model.fit(X_full, y_full)

    # Determine horizon length (at least 1 step)
    n = len(series_df)
    horizon_steps = max(1, int(n * horizon_fraction))

    # Prepare iterative forecasting state
    history_y = series_df.sort_values("ds")["y"].tolist()
    if len(history_y) < 7:
        raise ValueError("Need at least 7 observations to use lag7 feature")

    last_ds = series_df["ds"].max()

    # Cache last known exogenous feature values from the last available row in the training features
    last_feature_row = features.iloc[-1]
    last_exog_values: Dict[str, float] = {}
    for col in feature_cols:
        if col in ("lag1", "lag7") or col.startswith("dow_"):
            continue
        try:
            last_exog_values[col] = float(last_feature_row[col])
        except Exception:
            last_exog_values[col] = 0.0

    future_rows = []
    for step in range(1, horizon_steps + 1):
        current_ds = last_ds + pd.Timedelta(days=step)
        dow = int(current_ds.dayofweek)

        lag1 = float(history_y[-1])
        lag7 = float(history_y[-7]) if len(history_y) >= 7 else float("nan")

        # Build feature vector aligned with the training feature order
        row_values = []
        for col in feature_cols:
            if col == "lag1":
                row_values.append(lag1)
            elif col == "lag7":
                row_values.append(lag7)
            elif col.startswith("dow_"):
                # Column format is 'dow_X' where X in {0..6} (full one-hot)
                try:
                    col_dow = int(col.split("_")[1])
                except Exception:
                    col_dow = None
                row_values.append(1.0 if (col_dow is not None and dow == col_dow) else 0.0)
            else:
                # Exogenous features (numeric one-value, single-label one-hots, or multilabel indicators):
                # carry forward last known values from the training set
                row_values.append(float(last_exog_values.get(col, 0.0)))

        x = np.asarray(row_values, dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        history_y.append(yhat)
        future_rows.append({"ds": current_ds, "yhat": yhat})

    future_df = pd.DataFrame(future_rows)
    return future_df


def analyze_preprocessing(series_df: pd.DataFrame) -> Dict[str, object]:
    """
    Analyze preprocessing to surface counts for UI progress display.
    Returns a dict with keys:
      - lag_imputed_rows
      - num_dow_dummies
      - total_feature_columns
    """
    _, _, info = _build_features_internal(series_df, return_info=True)
    return info
