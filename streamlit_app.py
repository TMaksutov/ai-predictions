from pathlib import Path

import pandas as pd
import streamlit as st
from modeling import get_fast_estimators, train_full_and_forecast_future
from modeling import train_on_known_and_forecast_missing
from features import build_features as _build_features
from data_io import read_table_any as _read_table_any
from data_io import build_checklist_grouped
import io

st.set_page_config(page_title="Simple TS Benchmark", layout="wide")

# Enforce a wider sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 500px;
            max-width: 500px;
        }
        [data-testid="stSidebar"] > div:first-child {
            min-width: 500px;
            max-width: 500px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_PATH = Path(__file__).parent / "sample.csv"
METRIC_NAME = "MAPE"

# Utilities, I/O helpers, and checklist provided by: data_io.py, modeling.py, features.py

def run_benchmark(series_df: pd.DataFrame, test_fraction: float = 0.2):
    from modeling import benchmark_models
    return benchmark_models(series_df, test_fraction=test_fraction)


 # -----------------------------
 # UI
 # -----------------------------
# Single upload control
uploaded = st.sidebar.file_uploader(
    "Upload a CSV (date column first, target last)",
    type=["csv"],
    accept_multiple_files=False,
    help="If no file is uploaded, the default dataset 'sample.csv' will be used."
)

# Note: Keep raw_df/file_info/series/load_meta consistent. If an uploaded file has
# errors, do not fall back to default; show the checklist instead.
data_source_name = None
series = None
raw_df = pd.DataFrame()
file_info = {"error": None, "file_type": None, "n_rows": None, "n_cols": None, "separator": None, "header_detected": None, "header_names": None, "header_renamed_count": 0}
load_meta = {"original_time_col": None, "original_target_col": None, "trailing_missing_count": 0, "last_known_ds": None, "future_rows_raw": pd.DataFrame()}

if uploaded is not None:
    data_source_name = getattr(uploaded, "name", "uploaded.csv")
    try:
        raw_df, file_info = _read_table_any(uploaded)
    except Exception as e:
        file_info = {"error": f"Read failed: {e}", "file_type": None, "n_rows": 0, "n_cols": 0, "separator": None, "header_detected": None, "header_names": None, "header_renamed_count": 0}

    if not file_info.get("error"):
        try:
            # Build a clean series avoiding duplicate 'ds'/'y' labels regardless of original headers
            orig_cols = list(raw_df.columns)
            first_col = orig_cols[0]
            last_col = orig_cols[-1]

            ds_series = pd.to_datetime(raw_df[first_col], errors="coerce")
            y_series = pd.to_numeric(raw_df[last_col], errors="coerce")

            series = pd.DataFrame({
                "ds": ds_series,
                "y": y_series,
            })
            # Attach intermediate feature columns if any, skipping any that collide with 'ds'/'y'
            for c in orig_cols[1:-1]:
                if c in ("ds", "y"):
                    continue
                series[c] = raw_df[c]

            series = series.sort_values("ds").reset_index(drop=True)
            load_meta = {
                "original_time_col": first_col,
                "original_target_col": last_col,
                "trailing_missing_count": 0,
                "last_known_ds": series["ds"].max() if not series.empty else None,
                "future_rows_raw": pd.DataFrame(),
            }
        except Exception as e:
            file_info = {**file_info, "error": f"Build series failed: {e}"}
else:
    # No upload; use default
    default_path = SAMPLE_PATH
    data_source_name = default_path.name
    if default_path.exists():
        try:
            raw_df, file_info = _read_table_any(default_path)
        except Exception as e:
            file_info = {"error": f"Read failed: {e}", "file_type": None, "n_rows": 0, "n_cols": 0, "separator": None, "header_detected": None, "header_names": None, "header_renamed_count": 0}
        if not file_info.get("error"):
            try:
                # Build a clean series avoiding duplicate 'ds'/'y' labels for the default dataset as well
                orig_cols = list(raw_df.columns)
                first_col = orig_cols[0]
                last_col = orig_cols[-1]

                ds_series = pd.to_datetime(raw_df[first_col], errors="coerce")
                y_series = pd.to_numeric(raw_df[last_col], errors="coerce")

                series = pd.DataFrame({
                    "ds": ds_series,
                    "y": y_series,
                })
                for c in orig_cols[1:-1]:
                    if c in ("ds", "y"):
                        continue
                    series[c] = raw_df[c]

                series = series.sort_values("ds").reset_index(drop=True)
                load_meta = {
                    "original_time_col": first_col,
                    "original_target_col": last_col,
                    "trailing_missing_count": 0,
                    "last_known_ds": series["ds"].max() if not series.empty else None,
                    "future_rows_raw": pd.DataFrame(),
                }
            except Exception as e:
                file_info = {**file_info, "error": f"Build series failed: {e}"}
    else:
        file_info = {"error": f"Default data file not found: {default_path}", "file_type": None, "n_rows": 0, "n_cols": 0, "separator": None, "header_detected": None, "header_names": None, "header_renamed_count": 0}

if data_source_name:
    try:
        st.sidebar.caption(f"Dataset: {data_source_name}")
    except Exception:
        pass

# Sidebar checklist
progress_container = st.sidebar.container()
with progress_container:
    st.markdown("<div style='font-weight:600; margin:0 0 6px 0; text-align:center'>Checklist</div>", unsafe_allow_html=True)

def _render_checklist(grouped_items):
    with progress_container:
        for title, items in grouped_items:
            st.markdown(f"<div style='font-weight:600; margin:6px 0 3px 0'>{title}</div>", unsafe_allow_html=True)
            for idx, (status, text) in enumerate(items):
                icon = "✅" if status == "ok" else ("⚠️" if status == "warn" else "❌")
                st.markdown(f"<div style='margin:2px 0; line-height:1.2'>{icon} {text}</div>", unsafe_allow_html=True)

# Persisted UI defaults for display controls
for _key, _default in [
    ("show_only_test", True),
    ("hide_non_chosen_models", True),
    ("hide_table", True),
    ("hide_features_table", True),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# New user-facing toggles default to False (content hidden by default)
for _key, _default in [
    ("show_full_history", False),
    ("show_all_model_forecasts", False),
    ("show_model_comparison_table", False),
    ("show_training_data_preview", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Derive legacy flags from the new toggles so plotting logic uses them immediately
st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)
st.session_state["hide_non_chosen_models"] = not st.session_state.get("show_all_model_forecasts", False)
st.session_state["hide_table"] = not st.session_state.get("show_model_comparison_table", False)
st.session_state["hide_features_table"] = not st.session_state.get("show_training_data_preview", False)

# Read current control values from session state (widgets will render under the graph)
show_only_test = st.session_state.get("show_only_test", True)
hide_non_chosen_models = st.session_state.get("hide_non_chosen_models", True)
hide_table = st.session_state.get("hide_table", True)
hide_features_table = st.session_state.get("hide_features_table", True)

# Build checklist early and gate predictions/features if any non-OK in critical sections
# Initial checklist with no modeling artifacts
pre_grouped = build_checklist_grouped(
    raw_df,
    file_info,
    series if isinstance(series, pd.DataFrame) else pd.DataFrame(),
    load_meta,
    results=[],
    future_horizon=0,
)

def _has_blocking_issues(grouped_items):
    blocking_sections = {"Open & analyze", "Features & prep"}
    for title, items in grouped_items:
        if title not in blocking_sections:
            continue
        for status, _ in items:
            if status == "error":
                return True
    return False

if _has_blocking_issues(pre_grouped):
    _render_checklist(pre_grouped)
    st.stop()

try:
    if len(series) < 10:
        st.error("Insufficient data (need at least 10 rows) in the dataset.")
    else:
        # Single required mode: Predict missing targets at the end
        feature_cols = [c for c in series.columns if c not in ("ds", "y")]
        y_notna = series["y"].notna()
        if not y_notna.any():
            st.error("No known target values found. Provide history with targets and future rows with empty target.")
            st.stop()
        last_observed_idx = int(y_notna[y_notna].index.max())
        if last_observed_idx >= len(series) - 1:
            st.error("No future rows detected. Append future dates (and features if present) with empty target at the end.")
            st.stop()

        # Benchmark only on known rows
        series_for_benchmark = series.iloc[: last_observed_idx + 1].dropna(subset=["y"]).copy()
        results = run_benchmark(series_for_benchmark, test_fraction=0.2)

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
            best_name = sorted(results, key=lambda r: r.get("mape", float("inf")))[0]["name"]

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

        # Compute and plot predictions for the provided future rows
        future_horizon = None
        if best_name is not None:
            name_to_est = {name: est for name, est in get_fast_estimators()}
            ranked = sorted(results, key=lambda r: r.get("mape", float("inf")))
            chosen_res = None
            for res in ranked:
                if res["name"] in name_to_est:
                    chosen_res = res
                    break
            if chosen_res is not None:
                chosen_name = chosen_res["name"]
                # Reuse exact hyperparameters from benchmarking for retrain
                try:
                    best_params = chosen_res.get("estimator_params", {})
                    if best_params:
                        try:
                            name_to_est[chosen_name].set_params(**best_params)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Predict exactly the trailing missing targets using provided future rows (with features if present)
                future_rows = series.iloc[last_observed_idx + 1 :][["ds"] + feature_cols]
                future_df = train_on_known_and_forecast_missing(
                    series,
                    name_to_est[chosen_name],
                    future_rows=future_rows,
                )

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
        st.pyplot(fig)
        # Controls rendered under the graph (they update session_state and trigger rerun)
        controls_container = st.container()
        with controls_container:
            # New user-facing toggles propose showing additional content. Defaults are unchecked (False),
            # while underlying state keeps content hidden by default.
            st.checkbox(
                "Show full history",
                key="show_full_history",
                value=st.session_state.get("show_full_history", False)
            )
            st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)

            st.checkbox(
                "Show all model forecasts",
                key="show_all_model_forecasts",
                value=st.session_state.get("show_all_model_forecasts", False)
            )
            st.session_state["hide_non_chosen_models"] = not st.session_state.get("show_all_model_forecasts", False)

            st.checkbox(
                "Show model comparison table",
                key="show_model_comparison_table",
                value=st.session_state.get("show_model_comparison_table", False)
            )
            st.session_state["hide_table"] = not st.session_state.get("show_model_comparison_table", False)

            st.checkbox(
                "Show training data preview",
                key="show_training_data_preview",
                value=st.session_state.get("show_training_data_preview", False)
            )
            st.session_state["hide_features_table"] = not st.session_state.get("show_training_data_preview", False)

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
            "Model": r["name"], METRIC_NAME: r.get("mape", None),
            "Train (s)": r.get("train_time_s", None),
            "Predict (s)": r.get("predict_time_s", None)
        } for r in results]
        table_df = pd.DataFrame(table_rows).sort_values(METRIC_NAME).reset_index(drop=True)
        if not hide_table:
            st.markdown("#### Benchmark results")
            st.dataframe(table_df, use_container_width=True)

        # Render checklist in sidebar grouped by module
        try:
            file_info_local = locals().get('file_info', {})
            raw_df_local = locals().get('raw_df', pd.DataFrame())
            grouped = build_checklist_grouped(raw_df_local, file_info_local, series, load_meta, results, future_horizon or 0)
            _render_checklist(grouped)
        except Exception:
            pass

except Exception:
    # Suppress main-screen error banners; checklist already provides feedback
    _render_checklist(pre_grouped)
