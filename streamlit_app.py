from typing import List
from pathlib import Path

import pandas as pd
import streamlit as st
from models import get_fast_estimators, train_full_and_forecast_future
from models import analyze_preprocessing
from models import _build_features

st.set_page_config(page_title="Simple TS Benchmark", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "sample.csv"
METRIC_NAME = "nRMSE"


# -----------------------------
# Utilities
# -----------------------------
def load_series_from_csv(file_path: Path):
    """
    Assumes first column is timestamp and last column is target.
    Keeps all middle columns as exogenous features.
    Renames to (ds, y), coerces types, handles duplicates and daily reindex,
    filling y by interpolation and exogenous features appropriately.
    """
    import numpy as np

    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: time and target")

    first_col = df.columns[0]
    last_col = df.columns[-1]
    exog_cols_original = [c for c in df.columns if c not in [first_col, last_col]]

    # Rename key columns and keep all others
    df = df.rename(columns={first_col: "ds", last_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    rows_before = int(len(df))

    # Drop only rows with invalid/missing timestamps; keep y NaNs to impute later
    ds_valid_mask = df["ds"].notna()
    dropped_due_to_ds = int((~ds_valid_mask).sum())
    df = df.loc[ds_valid_mask].copy()

    # Handle duplicate timestamps: mean for y, last for exogenous
    duplicates_aggregated = int(df.duplicated(subset=["ds"]).sum())
    if duplicates_aggregated > 0:
        y_mean = df.groupby("ds", as_index=False)["y"].mean()
        last_vals = df.groupby("ds", as_index=False).last()
        last_vals = last_vals.drop(columns=["y"], errors="ignore")
        df = pd.merge(y_mean, last_vals, on="ds", how="left")
    df = df.sort_values("ds").reset_index(drop=True)
    unique_after_group = int(len(df))

    # Prepare daily index
    df = df.set_index("ds")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "ds"

    # Interpolate/Impute target y
    y_nan_before = int(df["y"].isna().sum())
    gaps_filled = int(df["y"].isna().sum())
    df["y"] = df["y"].interpolate(method="time")
    y_median_post_interp = df["y"].median()
    if pd.isna(y_median_post_interp):
        y_median_post_interp = float(df["y"].dropna().median() if not df["y"].dropna().empty else 0.0)
    df["y"] = df["y"].fillna(float(y_median_post_interp))
    y_interpolated_rows = int(gaps_filled)

    # Handle exogenous columns during reindex
    used_exog_cols = []
    numeric_exog_cols = []
    categorical_exog_cols = []
    for col in exog_cols_original:
        if col not in df.columns:
            # Column could be missing after groupby merge; skip
            continue
        series = df[col]
        # Try to coerce to numeric to detect numeric-like
        numeric_coerced = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric_coerced.notna().sum()) / float(len(series)) if len(series) > 0 else 0.0
        if numeric_ratio >= 0.6:  # treat as numeric if majority can be parsed
            df[col] = numeric_coerced
            # interpolate and fill median
            df[col] = df[col].interpolate(method="time")
            col_median = float(df[col].median()) if not np.isnan(df[col].median()) else 0.0
            df[col] = df[col].fillna(col_median)
            numeric_exog_cols.append(col)
            used_exog_cols.append(col)
        else:
            # Treat as categorical/text: forward fill then backfill, then 'Unknown'
            df[col] = series.fillna(method="ffill").fillna(method="bfill").fillna("Unknown")
            categorical_exog_cols.append(col)
            used_exog_cols.append(col)

    # Cap outliers on y using IQR method
    q1 = float(df["y"].quantile(0.25))
    q3 = float(df["y"].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    y_before_cap = df["y"].copy()
    df["y"] = df["y"].clip(lower=lower, upper=upper)
    y_capped_rows = int((y_before_cap != df["y"]).sum())

    sub = df.reset_index().rename(columns={"index": "ds"})
    rows_after = int(len(sub))

    meta = {
        "original_time_col": first_col,
        "original_target_col": last_col,
        "total_columns": int(len(["ds"] + used_exog_cols + ["y"])),
        "ignored_columns": 0,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": max(dropped_due_to_ds, 0),
        "duplicates_aggregated": int(duplicates_aggregated),
        "rows_added_by_resample": int(rows_after - unique_after_group),
        "y_interpolated_rows": int(y_interpolated_rows),
        "y_imputed_rows": int(y_nan_before),
        "y_capped_rows": int(y_capped_rows),
        "num_exog_numeric": int(len(numeric_exog_cols)),
        "num_exog_categorical": int(len(categorical_exog_cols)),
    }
    return sub, meta


def run_benchmark(series_df: pd.DataFrame, test_fraction: float = 0.2):
    from models import benchmark_models
    return benchmark_models(series_df, test_fraction=test_fraction)


 # -----------------------------
 # UI
 # -----------------------------
# Single upload control
uploaded = st.sidebar.file_uploader(
    "Upload a CSV (time column first, target last)",
    type=["csv"],
    accept_multiple_files=False,
    help="If no file is uploaded, the default dataset 'sample.csv' will be used."
)

# Determine data source: uploaded file or default
data_source_name = None
series = None

if uploaded is not None:
    try:
        series, load_meta = load_series_from_csv(uploaded)
        data_source_name = getattr(uploaded, "name", "uploaded.csv")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

if series is None:
    default_path = SAMPLE_PATH
    if not default_path.exists():
        st.error(f"Default data file not found: {default_path}")
        st.stop()
    try:
        series, load_meta = load_series_from_csv(default_path)
        data_source_name = default_path.name
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        st.stop()

# Sidebar progress checklist (simple)
progress_container = st.sidebar.container()
with progress_container:
    st.markdown("<div style='font-weight:600; margin:0 0 6px 0'>Progress</div>", unsafe_allow_html=True)
    success_icon, warn_icon, error_icon = "✅", "⚠️", "❌"
    ignored_cols = load_meta.get("ignored_columns", 0)
    rows_dropped = load_meta.get("rows_dropped", 0)
    rows_after = load_meta.get("rows_after", len(series))
    y_imputed_rows = load_meta.get("y_imputed_rows", 0)
    y_interpolated_rows = load_meta.get("y_interpolated_rows", 0)
    y_capped_rows = load_meta.get("y_capped_rows", 0)
    duplicates_aggregated = load_meta.get("duplicates_aggregated", 0)
    rows_added_by_resample = load_meta.get("rows_added_by_resample", 0)
    columns_icon = warn_icon if ignored_cols > 0 else success_icon
    missing_time_icon = warn_icon if rows_dropped > 0 else success_icon
    y_impute_icon = warn_icon if y_imputed_rows > 0 else success_icon
    loaded_icon = error_icon if rows_after == 0 else success_icon
    time_icon = success_icon
    line_style = "style='margin:2px 0; line-height:1.2'"
    st.markdown(f"<div {line_style}>{loaded_icon} Rows loaded: {rows_after}</div>", unsafe_allow_html=True)
    st.markdown(f"<div {line_style}>{time_icon} Time series column: {load_meta.get('original_time_col', 'unknown')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div {line_style}>{columns_icon} Columns detected: {load_meta.get('total_columns', 0)}</div>", unsafe_allow_html=True)
    dropped_icon = warn_icon if ignored_cols > 0 else success_icon
    st.markdown(f"<div {line_style}>{dropped_icon} Columns dropped: {ignored_cols}</div>", unsafe_allow_html=True)
    st.markdown(f"<div {line_style}>{missing_time_icon} Rows dropped (invalid time): {rows_dropped}</div>", unsafe_allow_html=True)
    agg_icon = success_icon if duplicates_aggregated == 0 else warn_icon
    st.markdown(f"<div {line_style}>{agg_icon} Duplicate timestamps aggregated: {duplicates_aggregated}</div>", unsafe_allow_html=True)
    resample_icon = success_icon if rows_added_by_resample == 0 else warn_icon
    st.markdown(f"<div {line_style}>{resample_icon} Rows added by daily resample: {rows_added_by_resample}</div>", unsafe_allow_html=True)
    interp_icon = success_icon if y_interpolated_rows == 0 else warn_icon
    st.markdown(f"<div {line_style}>{interp_icon} Target values interpolated: {y_interpolated_rows}</div>", unsafe_allow_html=True)
    st.markdown(f"<div {line_style}>{y_impute_icon} Target values imputed (median): {y_imputed_rows}</div>", unsafe_allow_html=True)
    cap_icon = success_icon if y_capped_rows == 0 else warn_icon
    st.markdown(f"<div {line_style}>{cap_icon} Outliers capped (IQR): {y_capped_rows}</div>", unsafe_allow_html=True)

# Controls next to the graph
controls_col, plot_col = st.columns([1, 5])

# Option to show only the last 20% (test window)
show_only_test = controls_col.checkbox("Hide train", value=True)

# Option to hide benchmark model lines except the chosen best model
hide_non_chosen_models = controls_col.checkbox("Hide models", value=True)

# Option to hide the benchmark results table
hide_table = controls_col.checkbox("Hide benchmark table", value=True)

# Option to hide the features sample table (first 5 rows)
hide_features_table = controls_col.checkbox("Hide features table", value=True)

# Run benchmark and plot
try:
    if len(series) < 30:
        st.error("Insufficient data (need at least 30 rows) in the dataset.")
    else:
        results = run_benchmark(series, test_fraction=0.2)

        import matplotlib.pyplot as plt

        # Plot all forecasts
        fig, ax = plt.subplots(figsize=(20, 6))

        # All models share same split; use first result for split line and test_df
        split_ds = None
        if len(results) > 0 and len(results[0]["test_df"]) > 0:
            split_ds = results[0]["test_df"]["ds"].iloc[0]

        # Determine best model name early for optional filtering of plotted forecasts
        best_name = None
        if len(results) > 0:
            best_name = sorted(results, key=lambda r: r["nrmse"])[0]["name"]

        # Optionally restrict actuals to only the test window
        visible_series = series
        if show_only_test and split_ds is not None:
            visible_series = series[series["ds"] >= split_ds]

        ax.plot(visible_series["ds"], visible_series["y"], linewidth=2, alpha=0.85, color="#222")

        for res in results:
            # Optionally skip non-chosen models from the benchmark forecasts
            if hide_non_chosen_models and best_name is not None and res["name"] != best_name:
                continue
            forecast_df = res["forecast_df"]
            label = res['name']
            # Prepend the last actual point before the forecast start to avoid a visual gap
            if not forecast_df.empty:
                first_forecast_ds = forecast_df["ds"].iloc[0]
                prev_actual = series[series["ds"] < first_forecast_ds].tail(1)
                if not prev_actual.empty:
                    pad_row = pd.DataFrame({
                        "ds": prev_actual["ds"].values,
                        "yhat": prev_actual["y"].values,
                    })
                    plot_df = pd.concat([pad_row, forecast_df], ignore_index=True)
                else:
                    plot_df = forecast_df
            else:
                plot_df = forecast_df

            ax.plot(plot_df["ds"], plot_df["yhat"], linestyle="--", linewidth=1.8, label=label)

        # Compute and plot future forecast (+20%) using the best benchmarked model
        future_horizon = None
        if best_name is not None:
            name_to_est = {name: est for name, est in get_fast_estimators()}
            if best_name in name_to_est:
                future_df = train_full_and_forecast_future(series, name_to_est[best_name], horizon_fraction=0.2)
                # Prepend the last actual point to the future forecast to avoid gap
                last_actual = series.sort_values("ds").tail(1)
                if not last_actual.empty:
                    pad_row = pd.DataFrame({
                        "ds": last_actual["ds"].values,
                        "yhat": last_actual["y"].values,
                    })
                    future_plot_df = pd.concat([pad_row, future_df], ignore_index=True)
                else:
                    future_plot_df = future_df

                ax.plot(
                    future_plot_df["ds"], future_plot_df["yhat"],
                    linestyle="-", linewidth=2.2, color="#d62728",
                    label=f"Prediction"
                )
                future_horizon = len(future_df)

        if split_ds is not None and not show_only_test:
            ax.axvline(split_ds, linestyle=":", linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout(rect=(0, 0, 0.8, 1))
        with plot_col:
            st.pyplot(fig)

        # Optional: show the dataframe used for training/prediction (first 5 rows)
        if not hide_features_table:
            try:
                features_df, feature_cols = _build_features(series)
                st.markdown("#### Training/prediction data (first 5 rows)")
                st.dataframe(features_df.head(5), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not build features preview: {e}")

        # Metrics table with timing
        table_rows = [{
            "Model": r["name"], METRIC_NAME: r["nrmse"],
            "Train (s)": r.get("train_time_s", None),
            "Predict (s)": r.get("predict_time_s", None)
        } for r in results]
        table_df = pd.DataFrame(table_rows).sort_values(METRIC_NAME).reset_index(drop=True)
        if not hide_table:
            st.markdown("#### Benchmark results")
            st.dataframe(table_df, use_container_width=True)

        # Update progress container (simple checklist continuation)
        with progress_container:
            success_icon, warn_icon, error_icon = "✅", "⚠️", "❌"
            # Best model and accuracy
            if len(results) > 0 and best_name is not None:
                best_result = sorted(results, key=lambda r: r["nrmse"])[0]
                best_nrmse = float(best_result.get("nrmse", 0.0))
                accuracy_pct = max(0.0, min(1.0, 1.0 - best_nrmse)) * 100.0
                st.markdown(f"<div {line_style}>{success_icon} Best model: {best_name}</div>", unsafe_allow_html=True)
                # Color-coded accuracy icon
                if accuracy_pct >= 80.0:
                    acc_icon = success_icon  # green
                elif accuracy_pct >= 60.0:
                    acc_icon = warn_icon    # yellow
                else:
                    acc_icon = error_icon   # red
                st.markdown(f"<div {line_style}>{acc_icon} Model accuracy: {accuracy_pct:.0f}%</div>", unsafe_allow_html=True)
            if future_horizon is not None:
                st.markdown(f"<div {line_style}>{success_icon} Point predicted: {future_horizon}</div>", unsafe_allow_html=True)

        # Preprocessing analysis (for feature engineering insight)
        try:
            prep_info = analyze_preprocessing(series)
        except Exception:
            prep_info = {}
except Exception as e:
    st.error(f"Failed to run forecast: {e}")
