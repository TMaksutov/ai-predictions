from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Import configuration
try:
    from utils.config import (
        LAG_PERIODS,
        MOVING_AVERAGE_WINDOWS,
        FOURIER_PERIODS,
        FOURIER_HARMONICS,
        AUTO_DETECT_FOURIER,
        AUTO_MAX_PERIOD,
        AUTO_TOP_N,
        AUTO_MIN_CYCLES,
    )
except ImportError:
    # Fallback values if config not available
    LAG_PERIODS = [1, 7, 30, 90, 365]
    MOVING_AVERAGE_WINDOWS = [7, 30, 90]
    FOURIER_PERIODS = [7, 30, 365]
    FOURIER_HARMONICS = 3
    AUTO_DETECT_FOURIER = True
    AUTO_MAX_PERIOD = 400
    AUTO_TOP_N = 3
    AUTO_MIN_CYCLES = 3

try:
    from utils.seasonality import detect_seasonal_periods
except Exception:
    def detect_seasonal_periods(y: np.ndarray, max_period: int = 400, top_n: int = 3, min_cycles: int = 3) -> List[int]:
        return []


# Note: Fourier terms functionality removed to simplify the feature engineering pipeline


def sanitize_feature_token(token: str) -> str:
    """Sanitize category/label tokens for safe column names."""
    if token is None:
        return "unknown"
    s = str(token)
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
    for delim in [",", ";", "|"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            return [p for p in parts if p]
    s = s.strip()
    if s and s.lower() != "nan":
        return [s]
    return []


def _is_probably_multilabel(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    for val in sample:
        labels = _parse_multilabel_cell(val)
        if len(labels) >= 2:
            return True
    return False


def build_features_internal(series_df: pd.DataFrame, return_info: bool = False) -> Tuple[pd.DataFrame, list, dict]:
    """
    Build regression features with lags, DOW one-hot, and exogenous handling.
    Returns (features_df, feature_columns, info_dict)
    """
    df = series_df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    # Respect data length when adding long lags / windows to avoid empty training after dropna
    n_rows = int(len(df))
    used_lags: List[int] = [int(l) for l in LAG_PERIODS if n_rows > int(l)]
    # Moving averages with min_periods=1 introduce at most one initial NaN after shift(1).
    # They are safe to include regardless of window size relative to data length.
    used_ma_windows: List[int] = [int(w) for w in MOVING_AVERAGE_WINDOWS]

    # Create lag features dynamically from configuration (filtered)
    for lag in used_lags:
        df[f"lag{lag}"] = df["y"].shift(lag)

    # Create moving average features dynamically from configuration (filtered)
    for window in used_ma_windows:
        df[f"ma{window}"] = df["y"].rolling(window=window, min_periods=1).mean().shift(1)

    # DOW one-hot
    df["dow"] = df["ds"].dt.dayofweek.astype(int)
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=False)

    # Global linear trend (days since start)
    df["trend"] = (df["ds"] - df["ds"].min()).dt.days.astype(int)

    # Fourier seasonality terms (fixed + optional auto-detected)
    fourier_cols: List[str] = []
    used_fourier_periods: List[int] = []
    try:
        base_periods = list(dict.fromkeys(int(p) for p in FOURIER_PERIODS if int(p) > 1))
    except Exception:
        base_periods = [7, 30, 365]

    detected_periods: List[int] = []
    if AUTO_DETECT_FOURIER:
        try:
            y_series = pd.to_numeric(df["y"], errors="coerce").dropna()
            detected = detect_seasonal_periods(
                y_series.to_numpy(dtype=float),
                max_period=int(AUTO_MAX_PERIOD),
                top_n=int(AUTO_TOP_N),
                min_cycles=int(AUTO_MIN_CYCLES),
            )
            detected_periods = [int(p) for p in detected if int(p) > 1]
        except Exception:
            detected_periods = []

    # Merge base and detected, dedupe and sort
    merged_periods: List[int] = []
    for p in base_periods + detected_periods:
        ip = int(p)
        if ip > 1 and ip not in merged_periods:
            merged_periods.append(ip)

    # Add Fourier terms using trend as time index
    if len(df) > 0 and FOURIER_HARMONICS and merged_periods:
        t = df["trend"].astype(float).to_numpy()
        for period in merged_periods:
            denom = float(period)
            if denom <= 0:
                continue
            for k in range(1, int(FOURIER_HARMONICS) + 1):
                angle = 2.0 * np.pi * float(k) * t / denom
                sin_col = f"fourier_sin_{period}_{k}"
                cos_col = f"fourier_cos_{period}_{k}"
                df[sin_col] = np.sin(angle)
                df[cos_col] = np.cos(angle)
                fourier_cols.extend([sin_col, cos_col])
        used_fourier_periods = merged_periods

    # Build base feature list dynamically
    base_features = ["ds", "y", "trend"]
    base_features.extend([f"lag{lag}" for lag in used_lags])
    base_features.extend([f"ma{window}" for window in used_ma_windows])

    base_block = df[base_features]
    features = pd.concat([base_block, dow_dummies], axis=1)
    # Attach Fourier columns if any
    if fourier_cols:
        features = pd.concat([features, df[fourier_cols]], axis=1)

    # Build feature columns list dynamically
    feature_cols = [f"lag{lag}" for lag in used_lags]
    feature_cols.extend([f"ma{window}" for window in used_ma_windows])
    feature_cols.extend(["trend"] + list(dow_dummies.columns))
    feature_cols.extend(list(fourier_cols))

    # Exogenous features - exclude time series features we've already created
    exog_exclude = {"dow", "trend"}
    exog_exclude.update([f"lag{lag}" for lag in LAG_PERIODS])
    exog_exclude.update([f"ma{window}" for window in MOVING_AVERAGE_WINDOWS])
    exog_cols = [c for c in df.columns if c not in ["ds", "y"] and c not in exog_exclude]
    num_exog_numeric = 0
    num_exog_categorical = 0
    num_exog_multilabel = 0
    exog_schema = []
    for col in exog_cols:
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(coerced.notna().sum()) / float(len(series)) if len(series) > 0 else 0.0
        if numeric_ratio >= 0.6:
            # Do not fill missing values; keep NaNs as-is
            ser_num = coerced
            new_name = f"x_{sanitize_feature_token(col)}"
            features[new_name] = ser_num.astype(float)
            feature_cols.append(new_name)
            num_exog_numeric += 1
            exog_schema.append({
                "original_name": col,
                "sanitized_base": sanitize_feature_token(col),
                "type": "numeric",
                "columns": [new_name],
            })
            continue

        if _is_probably_multilabel(series):
            parsed = series.apply(_parse_multilabel_cell)
            mlb = MultiLabelBinarizer()
            binarized = mlb.fit_transform(parsed)
            sanitized_base = sanitize_feature_token(col)
            binarized_cols = [f"m_{sanitized_base}_{sanitize_feature_token(c)}" for c in mlb.classes_]
            if binarized.shape[1] == 0:
                continue
            bin_df = pd.DataFrame(binarized, columns=binarized_cols, index=features.index)
            features = pd.concat([features, bin_df], axis=1)
            feature_cols.extend(binarized_cols)
            num_exog_multilabel += 1
            exog_schema.append({
                "original_name": col,
                "sanitized_base": sanitized_base,
                "type": "multilabel",
                "columns": binarized_cols,
                "classes": [str(c) for c in mlb.classes_],
            })
        else:
            # Do not fill missing values; include NaN as its own dummy
            ser_cat = series
            sanitized_base = sanitize_feature_token(col)
            dummies = pd.get_dummies(ser_cat, prefix=f"x_{sanitized_base}", drop_first=False, dummy_na=True)
            if not dummies.empty:
                features = pd.concat([features, dummies], axis=1)
                feature_cols.extend(list(dummies.columns))
                num_exog_categorical += 1
                exog_schema.append({
                    "original_name": col,
                    "sanitized_base": sanitized_base,
                    "type": "categorical",
                    "columns": list(dummies.columns),
                    # Store categories as strings without imputing missing
                    "categories": list(sorted({str(val) for val in ser_cat.dropna().unique()})),
                })

    info = {
        # No imputation performed
        "lag_imputed_rows": 0,
        "num_dow_dummies": int(len(dow_dummies.columns)),
        "total_feature_columns": int(len(feature_cols)),
        "num_exog_numeric": int(num_exog_numeric),
        "num_exog_categorical": int(num_exog_categorical),
        "num_exog_multilabel": int(num_exog_multilabel),
        "exog_schema": exog_schema,
        "used_lags": used_lags,
        "used_ma_windows": used_ma_windows,
        "fourier_periods_fixed": base_periods,
        "fourier_periods_detected": detected_periods,
        "fourier_harmonics": int(FOURIER_HARMONICS),
        "fourier_total_terms": int(len(fourier_cols)),
    }

    if return_info:
        return features.reset_index(drop=True), feature_cols, info
    return features.reset_index(drop=True), feature_cols, {}


def build_features(series_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    features, feature_cols, _ = build_features_internal(series_df, return_info=False)
    return features, feature_cols


def analyze_preprocessing(series_df: pd.DataFrame) -> Dict[str, object]:
    _, _, info = build_features_internal(series_df, return_info=True)
    return info


def select_base_feature_columns(all_columns: List[str]) -> List[str]:
    """Pick model features we can also recreate when forecasting.

    Includes: lags, moving averages, DOW dummies, trend, and exogenous features.
    """
    core_prefixes = ("lag", "ma", "dow_", "fourier_")
    exog_prefixes = ("x_", "m_")
    base_cols = [
        c
        for c in all_columns
        if c.startswith(core_prefixes) or c.startswith(exog_prefixes) or c == "trend"
    ]
    return base_cols


def build_base_row_for_date(
    base_cols: List[str],
    hist_df: pd.DataFrame,
    target_ds: pd.Timestamp,
    start_ds: pd.Timestamp,
) -> List[float]:
    """Create a feature row matching base_cols for a target date using history.

    - hist_df contains only known/previously predicted 'y' values with ascending 'ds'.
    - start_ds is the first timestamp of the original series for computing trend.
    """
    values: List[float] = []
    trend_val = int((pd.Timestamp(target_ds) - pd.Timestamp(start_ds)).days)
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
        if col.startswith("fourier_sin_") or col.startswith("fourier_cos_"):
            try:
                parts = col.split("_")
                # Format: fourier_sin_{period}_{k} or fourier_cos_{period}_{k}
                period = float(parts[2])
                k = float(parts[3])
                angle = 2.0 * np.pi * k * float(trend_val) / max(period, 1.0)
                if col.startswith("fourier_sin_"):
                    values.append(float(np.sin(angle)))
                else:
                    values.append(float(np.cos(angle)))
            except Exception:
                values.append(0.0)
            continue
        if col.startswith("lag"):
            try:
                lag_k = int(col.replace("lag", ""))
            except Exception:
                lag_k = 1
            idx = -lag_k
            val = float(y_series.iloc[idx]) if len(y_series) >= abs(idx) else float(y_series.iloc[-1])
            values.append(val)
            continue
        if col.startswith("ma"):
            try:
                window = int(col.replace("ma", ""))
            except Exception:
                window = 7
            window = max(1, window)
            tail = y_series.tail(window)
            values.append(float(tail.mean()))
            continue
        if col.startswith("x_") or col.startswith("m_"):
            # Exogenous placeholders (if present in base_cols, will be appended separately via schema)
            values.append(0.0)
            continue
        values.append(0.0)
    return values


def build_exog_vector_for_date(
    target_ds: pd.Timestamp,
    future_map: Dict,
    exog_schema: List[Dict],
    *,
    fill_missing_numeric_with_zero: bool = True,
) -> List[float]:
    """Construct exogenous feature vector for a given date using training schema.

    - future_map maps ds -> original (non-engineered) exogenous values
    - exog_schema contains the engineered columns per original exog feature
    """
    values: List[float] = []
    raw = future_map.get(pd.Timestamp(target_ds), {})
    for sch in exog_schema:
        typ = sch.get("type")
        cols = list(sch.get("columns", []))
        if typ == "numeric":
            original_name = sch.get("original_name")
            val = pd.to_numeric(raw.get(original_name, pd.NA), errors="coerce")
            if pd.isna(val):
                values.append(0.0 if fill_missing_numeric_with_zero else np.nan)  # type: ignore[arg-type]
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
            labels: List[str] = []
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