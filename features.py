from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def _fourier_terms_from_trend(trend: pd.Series, period: int, order: int, prefix: str) -> pd.DataFrame:
    """Create Fourier series terms using an integer day trend index.

    Columns are named like: f"{prefix}_p{period}_k{order}_{sin|cos}".
    """
    cols: Dict[str, List[float]] = {}
    t = trend.astype(float)
    for k in range(1, int(order) + 1):
        angle = 2.0 * np.pi * k * (t / float(period))
        cols[f"fourier_p{int(period)}_k{k}_sin"] = np.sin(angle)
        cols[f"fourier_p{int(period)}_k{k}_cos"] = np.cos(angle)
    return pd.DataFrame(cols, index=trend.index)


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

    # Lags (no filling; preserve missing values)
    df["lag1"] = df["y"].shift(1)
    df["lag7"] = df["y"].shift(7)
    df["lag14"] = df["y"].shift(14)
    df["lag28"] = df["y"].shift(28)

    # Rolling means (preserve missing values for early rows; shift to avoid leakage)
    df["ma7"] = df["y"].rolling(window=7, min_periods=1).mean().shift(1)
    df["ma14"] = df["y"].rolling(window=14, min_periods=1).mean().shift(1)
    df["ma28"] = df["y"].rolling(window=28, min_periods=1).mean().shift(1)

    # DOW one-hot
    df["dow"] = df["ds"].dt.dayofweek.astype(int)
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=False)

    # Global linear trend (days since start)
    df["trend"] = (df["ds"] - df["ds"].min()).dt.days.astype(int)

    # Fourier seasonality terms
    # - Weekly (7)
    # - Monthly (30) and Quarterly (90) to capture sub-annual cycles
    # - Yearly (365) with higher order for richer seasonality
    # - Biannual (730) for slower cycles observed in some notebooks
    fourier_weekly = _fourier_terms_from_trend(df["trend"], period=7, order=3, prefix="fourier")
    fourier_monthly = _fourier_terms_from_trend(df["trend"], period=30, order=3, prefix="fourier")
    fourier_quarterly = _fourier_terms_from_trend(df["trend"], period=90, order=2, prefix="fourier")
    fourier_yearly = _fourier_terms_from_trend(df["trend"], period=365, order=6, prefix="fourier")
    fourier_biannual = _fourier_terms_from_trend(df["trend"], period=730, order=2, prefix="fourier")

    base_block = df[[
        "ds",
        "y",
        "lag1",
        "lag7",
        "lag14",
        "lag28",
        "ma7",
        "ma14",
        "ma28",
        "trend",
    ]]
    features = pd.concat([
        base_block,
        dow_dummies,
        fourier_weekly,
        fourier_monthly,
        fourier_quarterly,
        fourier_yearly,
        fourier_biannual,
    ], axis=1)
    feature_cols = [
        "lag1",
        "lag7",
        "lag14",
        "lag28",
        "ma7",
        "ma14",
        "ma28",
        "trend",
    ] + list(dow_dummies.columns) \
        + list(fourier_weekly.columns) \
        + list(fourier_monthly.columns) \
        + list(fourier_quarterly.columns) \
        + list(fourier_yearly.columns) \
        + list(fourier_biannual.columns)

    # Exogenous
    exog_exclude = {
        "lag1", "lag7", "lag14", "lag28",
        "ma7", "ma14", "ma28",
        "dow", "trend",
    }
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


