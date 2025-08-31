from pathlib import Path
import sys

# Ensure project root is on sys.path for module imports in various runtimes
_APP_DIR = Path(__file__).parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
import streamlit as st
from modeling import get_fast_estimators
from modeling import train_on_known_and_forecast_missing, UnifiedTimeSeriesTrainer
from features import build_features as _build_features
from data_io import load_data_with_checklist
from data_utils import load_default_dataset, prepare_series_from_dataframe, get_future_rows
from plot_utils import create_forecast_plot, create_results_table
from trend import fit_trend
import io
import uuid
import json
import os
import time
from urllib import request
import warnings
from sklearn.exceptions import ConvergenceWarning

st.set_page_config(
    page_title="Daily Data Forecast",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress noisy optimization convergence warnings from scikit-learn in UI
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Enforce a wider sidebar and remove top padding
st.markdown(
    """
    <style>
        .block-container { padding: 1rem; margin: 1rem; }
        header[data-testid="stHeader"] { height: 20px; }
        [data-testid="stSidebar"] { min-width: 450px; max-width: 450px; }
        [data-testid="stSidebar"] > div:first-child { min-width: 450px; max-width: 450px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_PATH = Path(__file__).parent / "sample.csv"
METRIC_NAME = "RMSE"

# Utilities, I/O helpers, and checklist provided by: data_io.py, modeling.py, features.py

def _df_fingerprint(df: pd.DataFrame) -> str:
    """Create a stable fingerprint for a dataframe to use as a cache key."""
    try:
        # Sum of hash objects is deterministic for the same content in a session
        return str(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        # Fallback to a coarse fingerprint
        try:
            ds_min = pd.to_datetime(df["ds"]).min()
            ds_max = pd.to_datetime(df["ds"]).max()
            count_y = int(df["y"].notna().sum()) if "y" in df.columns else 0
            return f"shape={df.shape}|{ds_min}|{ds_max}|y_count={count_y}"
        except Exception:
            return f"shape={df.shape}"


def _ga_track(event_name, params=None):
    """Send a minimal, anonymous GA4 Measurement Protocol event.

    Requires secrets (or env vars): GA_MEASUREMENT_ID, GA_API_SECRET
    """
    try:
        mid = st.secrets.get("GA_MEASUREMENT_ID") or os.environ.get("GA_MEASUREMENT_ID")
        sec = st.secrets.get("GA_API_SECRET") or os.environ.get("GA_API_SECRET")
        if not mid or not sec:
            return
        if "ga_client_id" not in st.session_state:
            st.session_state["ga_client_id"] = str(uuid.uuid4())
        payload = {
            "client_id": st.session_state["ga_client_id"],
            "events": [{"name": str(event_name), "params": params or {}}],
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={mid}&api_secret={sec}"
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        request.urlopen(req, timeout=2)
    except Exception:
        # Never raise in UI; analytics is best-effort only
        pass

def run_benchmark(series_df: pd.DataFrame, test_fraction: float = None):
    if test_fraction is None:
        try:
            from config import DEFAULT_TEST_FRACTION
            test_fraction = DEFAULT_TEST_FRACTION
        except ImportError:
            test_fraction = 0.2

    trainer = UnifiedTimeSeriesTrainer()
    return trainer.benchmark_models(series_df, test_fraction=test_fraction)


# Helper functions for model selection
def _base_model_type(model_name: str) -> str:
    """Extract base type from a model name by trimming trailing digits."""
    try:
        # Normalize name: strip spaces and remove optional "+Trend" suffix
        model_name = str(model_name).replace("+Trend", "").strip()
        for idx, ch in enumerate(model_name):
            if ch.isdigit():
                return model_name[:idx] or model_name
        return model_name
    except Exception:
        return str(model_name)


def _pick_top_models(results: list, max_models: int = 3) -> list:
    """Pick the top-N models by RMSE (lower is better)."""
    try:
        # Sort by RMSE in ascending order (lowest RMSE first)
        ordered = sorted(results, key=lambda x: x.get("rmse", float('inf')))
        names = [r.get("name") for r in ordered if r.get("name")]
        return names[:max_models]
    except Exception:
        return []


 # -----------------------------
# UI
# -----------------------------

# Sidebar header
st.sidebar.markdown("<div style='font-weight:600; margin:6px 0 12px 0; text-align:center; font-size:18px'>Upload Daily Data</div>", unsafe_allow_html=True)

# Data loading - simplified using utility functions
uploaded = st.sidebar.file_uploader(
    "Upload a CSV or Excel file (date column first, target last, < 1MB)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    help="If no file is uploaded, the default dataset 'sample.csv' will be used. Maximum file size: 1MB; maximum rows: 10,000"
)

# Auto-load sample as a pseudo-upload on first load so it follows the same path
if uploaded is None:
    try:
        with open(str(SAMPLE_PATH), "rb") as _fb:
            _sample_bytes = _fb.read()
        _auto_file = io.BytesIO(_sample_bytes)
        _auto_file.name = SAMPLE_PATH.name
        _auto_file.size = len(_sample_bytes)
        uploaded = _auto_file
    except Exception:
        pass

# Privacy note ( upload counts only; no personal data)
st.sidebar.caption("This app collects  upload counts only. No personal data.")

 





# Sidebar checklist
progress_container = st.sidebar.container()
with progress_container:
    st.markdown("<div style='font-weight:600; margin:0 0 6px 0; text-align:center'>Checklist</div>", unsafe_allow_html=True)

# Remove progressive callback functionality - let checklist be shown at the end only
def update_checklist_callback(checklist_items):
    """Callback disabled - checklist will be shown at the end"""
    pass  # Do nothing during progressive updates

# Load data using unified logic (uploaded or auto-loaded sample as uploaded)
if uploaded is not None:
    data_source_name = getattr(uploaded, "name", "uploaded.csv")
    raw_df, file_info = load_data_with_checklist(uploaded, progress_callback=update_checklist_callback)
    # Count a single upload event per user session; no PII sent
    try:
        if not st.session_state.get("ga_uploaded_once", False):
            _ga_track("file_uploaded")
            st.session_state["ga_uploaded_once"] = True
    except Exception:
        pass
else:
    st.error("No file uploaded and sample could not be loaded.")
    st.stop()

# Prepare standardized series
series, load_meta = prepare_series_from_dataframe(raw_df, file_info)
try:
    # Enforce the detected date format consistently after checks
    detected_fmt = file_info.get("detected_date_format") if isinstance(file_info, dict) else None
    if not pd.api.types.is_datetime64_any_dtype(series["ds"]):
        if detected_fmt:
            series["ds"] = pd.to_datetime(series["ds"], errors="coerce", format=str(detected_fmt))
        else:
            series["ds"] = pd.to_datetime(series["ds"], errors="coerce", dayfirst=True)
    series["ds"] = series["ds"].dt.normalize()
except Exception:
    pass

# Dataset name caption removed per user request

def _render_checklist(items):
    with progress_container:
        st.markdown(f"<div style='font-weight:600; margin:6px 0 3px 0'>Validation Checklist</div>", unsafe_allow_html=True)
        for status, text in items:
            icon = "✅" if status == "ok" else ("⚠️" if status == "warn" else "❌")
            st.markdown(f"<div style='margin:2px 0; line-height:1.2'>{icon} {text}</div>", unsafe_allow_html=True)

# Persisted UI defaults for display controls
for _key, _default in [
    ("show_only_test", True),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# New user-facing toggles default to sensible values
for _key, _default in [
    ("show_full_history", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Keep legacy flag in sync
st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)

# Read current control values from session state (widgets will render under the graph)
show_only_test = st.session_state.get("show_only_test", True)

# Display the complete checklist after loading
if "checklist" in file_info and file_info["checklist"]:
    with progress_container:
        for status, text in file_info["checklist"]:
            icon = "✅" if status == "ok" else ("⚠️" if status == "warning" else "❌")
            st.markdown(f"<div style='margin:2px 0; line-height:1.2'>{icon} {text}</div>", unsafe_allow_html=True)

# Check validation status
if file_info.get("error"):
    # There was a loading error
    st.stop()

# If we got here, validation passed
is_valid = len(raw_df) > 0 and not file_info.get("error")

 

# Main screen header
st.markdown("<div style='font-weight:600; margin:12px 0 18px 0; text-align:center; font-size:24px'>Prediction Models Benchmark and Forecast</div>", unsafe_allow_html=True)

try:
    if len(series) < 10:
        st.error("Insufficient data (need at least 10 rows) in the dataset.")
    elif len(series.columns) < 2:
        st.error("Dataset must have at least 2 columns: date column (first) and target column (last). Your file appears to have only 1 column. Please check your CSV format - make sure it uses commas as separators.")
    else:
        # Single required mode: Predict missing targets at the end
        feature_cols = [c for c in series.columns if c not in ("ds", "y")]
        y_notna = series["y"].notna()
        if not y_notna.any():
            st.error("No known target values found. Provide history with targets and future rows with empty target.")
            st.stop()

        future_rows, last_observed_idx = get_future_rows(series, feature_cols)
        
        # Allow continuing even when no future rows - show training comparison table
        has_future_predictions = not future_rows.empty

        # Benchmark only on known rows
        series_for_benchmark = series.iloc[: last_observed_idx + 1].dropna(subset=["y"]).copy()
        # Import config for default values
        try:
            from config import DEFAULT_TEST_FRACTION
        except ImportError:
            DEFAULT_TEST_FRACTION = 0.2

        # Compute trend description early for use throughout the function
        _trend_desc = None
        try:
            known = series.dropna(subset=["y"]).copy()
            if not known.empty:
                from trend import fit_trend
                tm = fit_trend(known)
                deg = int(getattr(tm, "degree", 0))
                if deg <= 0:
                    _trend_desc = "No trend detected"
                elif deg == 1:
                    _trend_desc = "Linear trend"
                else:
                    _trend_desc = "Linear+Parabola trend"
        except Exception:
            _trend_desc = None

        try:
            trainer = UnifiedTimeSeriesTrainer()
            results = trainer.benchmark_models(series_for_benchmark, test_fraction=DEFAULT_TEST_FRACTION)
        except Exception as e:
            results = []
            st.error(f"Benchmark failed: {e}")
            # Fallback: provide a few baseline models so future forecasting can proceed
            try:
                baseline = [name for name, _ in get_fast_estimators()][:3]
            except Exception:
                baseline = []
            if baseline:
                results = [
                    {"name": n, "rmse": float("inf"), "mape": None, "accuracy_pct": 0.0,
                     "forecast_df": pd.DataFrame(), "test_df": pd.DataFrame(),
                     "train_time_s": None, "predict_time_s": None,
                     "estimator_params": {}}
                    for n in baseline
                ]

        # Get split date for plotting
        split_ds = None
        if len(results) > 0 and len(results[0]["test_df"]) > 0:
            split_ds = results[0]["test_df"]["ds"].iloc[0]

        # Get best model name
        best_name = None
        if len(results) > 0:
            best_name = sorted(results, key=lambda r: r.get("rmse", float("inf")))[0]["name"]

        # Determine model visibility (per-row checkboxes) and compute future predictions only for selected
        future_df = None
        future_horizon = 0
        selected_models = set()
        # Placeholders to control layout order: plot above, table below
        plot_container = st.container()
        table_container = st.container()
        # Compute top 3 models by RMSE (lower is better)
        top_models = _pick_top_models(results, max_models=3)
        top_models_set = set(top_models)
        # top_index_set is no longer needed since we use top_models_set based on model names
        if best_name is not None:
            # Initialize or reconcile visibility state with separate test/prediction controls
            available_models = [r["name"] for r in results]
            
            # Create a data fingerprint to detect when data changes (which affects top models)
            data_fingerprint = _df_fingerprint(series)
            current_data_key = st.session_state.get("current_data_key")
            
            # Force reset visibility state when data changes (new file, different dataset)
            # This ensures that when switching between datasets or after code changes,
            # the top 3 models by RMSE are always selected by default instead of
            # persisting old selections from previous runs
            if current_data_key != data_fingerprint:
                st.session_state["test_visibility"] = None
                st.session_state["pred_visibility"] = None
                st.session_state["current_data_key"] = data_fingerprint
            
            test_visibility = st.session_state.get("test_visibility")
            pred_visibility = st.session_state.get("pred_visibility")
            
            # Initialize test visibility (all models can be shown for test data)
            if (not isinstance(test_visibility, dict)) or (set(test_visibility.keys()) != set(available_models)):
                test_visibility = {name: (name in top_models_set) for name in available_models}
                st.session_state["test_visibility"] = test_visibility
            
            # Initialize prediction visibility (only top 3 models by RMSE can be shown for predictions)
            if (not isinstance(pred_visibility, dict)) or (set(pred_visibility.keys()) != set(available_models)):
                pred_visibility = {name: (name in top_models_set) for name in available_models}
                st.session_state["pred_visibility"] = pred_visibility

            # Apply any pending edits from the editor state so toggles reflect immediately on rerun
            try:
                editor_state = st.session_state.get("model_table_editor", None)
                if isinstance(editor_state, dict) and isinstance(editor_state.get("edited_rows"), dict):
                    index_to_model = [r["name"] for r in results]
                    for idx, changes in editor_state["edited_rows"].items():
                        try:
                            if 0 <= idx < len(index_to_model):
                                model_name = index_to_model[idx]
                                if "Show" in changes:
                                    show_checked = bool(changes["Show"])
                                    test_visibility[model_name] = show_checked
                                    if model_name in top_models_set:
                                        pred_visibility[model_name] = show_checked
                                    else:
                                        pred_visibility[model_name] = False
                        except Exception:
                            pass
                    st.session_state["test_visibility"] = test_visibility
                    st.session_state["pred_visibility"] = pred_visibility
            except Exception:
                pass

            # Ensure future predictions are computed before building the table so Predict (s) is populated
            if has_future_predictions:
                try:
                    ser_ds = future_rows["ds"]
                    if pd.api.types.is_datetime64_any_dtype(ser_ds):
                        key_vals = ser_ds.dt.strftime('%Y-%m-%d').tolist()
                    else:
                        key_vals = pd.to_datetime(ser_ds, errors="coerce").dt.strftime('%Y-%m-%d').tolist()
                    fr_ds_key = ",".join(key_vals)
                except Exception:
                    fr_ds_key = str(len(future_rows))
                future_key = f"{_df_fingerprint(series)}|future={fr_ds_key}"
                future_cache = st.session_state.get("future_cache", {})

                if future_key not in future_cache:
                    try:
                        with st.spinner("Forecasting future values..."):
                            reg_estimators = list(get_fast_estimators())
                            name_to_est_full = {name: est for name, est in reg_estimators}
                            base_to_est = {}
                            for disp_name, est in reg_estimators:
                                base = str(disp_name).split(" (")[0]
                                if base not in base_to_est:
                                    base_to_est[base] = est
                            model_to_future = {}
                            model_to_future_time = {}
                            for idx, res in enumerate(results):
                                model_name = res.get("name")
                                if model_name not in top_models_set:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    model_to_future_time[model_name] = None
                                    continue
                                is_trend_model = "+Trend" in model_name
                                base_model_name = model_name.replace("+Trend", "") if is_trend_model else model_name
                                est = base_to_est.get(base_model_name) or name_to_est_full.get(base_model_name)
                                if est is None:
                                    for disp_name, cand in reg_estimators:
                                        if str(disp_name).startswith(base_model_name):
                                            est = cand
                                            break
                                if est is None:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    model_to_future_time[model_name] = None
                                    continue
                                params = res.get("estimator_params", {})
                                try:
                                    if params:
                                        est.set_params(**params)
                                except Exception:
                                    pass
                                try:
                                    t_start_future = time.time()
                                    if is_trend_model:
                                        from trend import fit_trend
                                        known_data = series.dropna(subset=["y"]).copy()
                                        trend_model = fit_trend(known_data)
                                        detrended_series = series.copy()
                                        trend_fitted = trend_model.fitted(detrended_series["ds"])
                                        detrended_series.loc[detrended_series["y"].notna(), "y"] = (
                                            detrended_series.loc[detrended_series["y"].notna(), "y"] - 
                                            trend_fitted[detrended_series["y"].notna()].values
                                        )
                                        res_future_df = train_on_known_and_forecast_missing(
                                            detrended_series, est, future_rows=future_rows
                                        )
                                        if not res_future_df.empty:
                                            future_trend = trend_model.extrapolate(res_future_df["ds"])
                                            res_future_df["yhat"] = res_future_df["yhat"] + future_trend.values
                                    else:
                                        res_future_df = train_on_known_and_forecast_missing(
                                            series, est, future_rows=future_rows
                                        )
                                    t_end_future = time.time()
                                    if res_future_df.empty:
                                        res_future_df = pd.DataFrame({
                                            "ds": future_rows["ds"],
                                            "yhat": [series["y"].dropna().iloc[-1]] * len(future_rows)
                                        })
                                    model_to_future_time[model_name] = float(max(0.0, t_end_future - t_start_future))
                                except Exception:
                                    res_future_df = pd.DataFrame({
                                        "ds": future_rows["ds"],
                                        "yhat": [series["y"].dropna().iloc[-1] if series["y"].notna().any() else 0.0] * len(future_rows)
                                    })
                                    model_to_future_time[model_name] = None
                                model_to_future[model_name] = res_future_df
                            future_cache[future_key] = {"preds": model_to_future, "times": model_to_future_time}
                            st.session_state["future_cache"] = future_cache
                    except Exception:
                        pass

                # Attach cached results immediately so Predict (s) is available in the table
                cache_entry = future_cache.get(future_key, {})
                if isinstance(cache_entry, dict) and ("preds" in cache_entry or "times" in cache_entry):
                    model_to_future = cache_entry.get("preds", {})
                    model_to_future_time = cache_entry.get("times", {})
                else:
                    model_to_future = cache_entry if isinstance(cache_entry, dict) else {}
                    model_to_future_time = {k: None for k in model_to_future.keys()}
                for idx, res in enumerate(results):
                    model_name = res.get("name")
                    if model_name in top_models_set:
                        res["future_df"] = model_to_future.get(model_name, pd.DataFrame({"ds": [], "yhat": []}))
                        res["predict_future_time_s"] = model_to_future_time.get(model_name, None)
                    else:
                        res["future_df"] = pd.DataFrame({"ds": [], "yhat": []})
                        res["predict_future_time_s"] = None

            # Table will render below the plot

            # Use the new average-based accuracy calculation from modeling
            def _calculate_accuracy_for_plot(r):
                accuracy_pct = r.get('accuracy_pct', 0.0)
                if pd.notna(accuracy_pct) and accuracy_pct is not None:
                    return max(0.0, min(100.0, float(accuracy_pct)))
                else:
                    return 0.0
            
            # Create a compact benchmark table using st.data_editor
            def _fmt(v):
                try:
                    import math
                    if v is None:
                        return ""
                    if v == "N/A":
                        return "N/A"
                    if hasattr(v, 'item'):
                        v = v.item()
                    if isinstance(v, float):
                        return f"{v:.3f}" if math.isfinite(v) else ""
                    if isinstance(v, int):
                        return f"{float(v):.3f}"
                    return str(v)
                except Exception:
                    return str(v) if v is not None else ""

            table_data = []
            for r in results:
                m = r["name"]
                test_checked = bool(test_visibility.get(m, m in top_models_set))
                # Only allow prediction checkbox for top 3 models by RMSE
                pred_checked = bool(pred_visibility.get(m, False)) if m in top_models_set else False
                # Use the new average-based accuracy calculation from modeling
                accuracy_pct = r.get('accuracy_pct', 0.0)
                if pd.notna(accuracy_pct) and accuracy_pct is not None:
                    acc_pct = max(0.0, min(100.0, float(accuracy_pct)))
                else:
                    acc_pct = 0.0
                
                # Model display name (no extra suffix)
                model_display = m
                # Simple checkbox - if either test or prediction is checked, show the checkbox as checked
                show_checked = test_checked or pred_checked
                
                # predict_time_s from benchmarks is actually test prediction time, so we'll use it for Test (s)
                # For future prediction time, show the measured seconds for top-3 models by accuracy
                future_predict_time = r.get("predict_future_time_s", None) if m in top_models_set else ""
                
                table_data.append({
                    "Show": show_checked,
                    "Model": model_display,
                    f"{METRIC_NAME}": _fmt(r.get('rmse', None)),
                    "Accuracy (%)": _fmt(acc_pct),
                    "Train (s)": _fmt(r.get('train_time_s', None)),
                    "Test (s)": _fmt(r.get('predict_time_s', None)),  # Time to predict on test data during benchmarking
                    "Predict (s)": _fmt(future_predict_time)  # Time to predict future values (only for top 3 models by RMSE)
                })

            results_df = pd.DataFrame(table_data)

            with table_container:
                # Create two columns: table on the left, scatter plot on the right
                col1, col2 = st.columns([2, 1])  # Table takes 2/3 width, plot takes 1/3
                
                with col1:
                    # Create column configuration
                    column_config = {
                        "Show": st.column_config.CheckboxColumn(
                            "Show",
                            help="Display this model's test and prediction lines (if available)",
                            default=False,
                        ),
                        "Model": st.column_config.TextColumn(
                            "Model",
                            disabled=True,
                        ),
                        f"{METRIC_NAME}": st.column_config.TextColumn(
                            f"{METRIC_NAME}",
                            disabled=True,
                        ),
                        "Accuracy (%)": st.column_config.TextColumn(
                            "Accuracy (%)",
                            help="Accuracy based on mean absolute error relative to test dataset range. 99% means predictions are off by 1% of the test dataset value range.",
                            disabled=True,
                        ),
                        "Train (s)": st.column_config.TextColumn(
                            "Train (s)",
                            disabled=True,
                        ),
                        "Test (s)": st.column_config.TextColumn(
                            "Test (s)",
                            help="Time to predict on test data during benchmarking",
                            disabled=True,
                        ),
                                            "Predict (s)": st.column_config.TextColumn(
                        "Predict (s)",
                        help="Time to predict future values (only computed for top 3 models by RMSE)",
                        disabled=True,
                    ),
                    }
                    
                    # For models not in top 3 by accuracy, disable the prediction checkbox by making it read-only
                    # We'll handle this in the processing logic instead of using disabled parameter
                    edited_df = st.data_editor(
                        results_df,
                        hide_index=True,
                        column_config=column_config,
                        key="model_table_editor"
                    )
                
                with col2:
                    # Small scatter plot: Accuracy vs Time
                    try:
                        df = pd.DataFrame([
                            {
                                "Model": (n := r.get("name", "")),
                                "ShortName": str(n).split(" ")[0],
                                "Accuracy (%)": _calculate_accuracy_for_plot(r),
                                "Time (s)": float(r.get("train_time_s", 0.0) or 0.0) + float(r.get("predict_time_s", 0.0) or 0.0),
                                "Top3": n in top_models_set,
                            }
                            for r in results
                        ]).dropna(subset=["Accuracy (%)"]) 
                        if not df.empty:
                            st.vega_lite_chart(
                                df,
                                {
                                    "width": 400, "height": 400,  # Smaller size to fit in column
                                    "layer": [
                                        {
                                            "mark": {"type": "point", "filled": True, "size": 60},
                                            "encoding": {
                                                "x": {"field": "Time (s)", "type": "quantitative", "title": "Time (s)"},
                                                "y": {"field": "Accuracy (%)", "type": "quantitative", "title": "Accuracy (%)", "scale": {"zero": False}},
                                                "color": {"field": "Top3", "type": "nominal", "scale": {"domain": [True, False], "range": ["#d62728", "#bbbbbb"]}, "legend": None},
                                                "tooltip": [
                                                    {"field": "Model", "type": "nominal"},
                                                    {"field": "Accuracy (%)", "type": "quantitative", "format": ".3f"},
                                                    {"field": "Time (s)", "type": "quantitative", "format": ".1f"}
                                                ]
                                            }
                                        },
                                        {
                                            "transform": [{"filter": "datum.Top3"}],
                                            "mark": {"type": "text", "align": "left", "dx": 6, "dy": -6, "fontSize": 11, "color": "#d62728"},
                                            "encoding": {
                                                "x": {"field": "Time (s)", "type": "quantitative"},
                                                "y": {"field": "Accuracy (%)", "type": "quantitative", "scale": {"zero": False}},
                                                "text": {"field": "ShortName", "type": "nominal"}
                                            }
                                        }
                                    ]
                                },
                                use_container_width=False,
                            )
                    except:
                        pass

            for _, row in edited_df.iterrows():
                model_display = row["Model"]
                # Extract real model name (remove " (no predictions)" suffix if present)
                model_name = model_display.replace(" (no predictions)", "")
                show_checked = bool(row["Show"])
                
                # Simple logic: if checked, show both test and predictions (if available)
                test_visibility[model_name] = show_checked
                
                # Only show predictions for top 3 models by RMSE, and only if checked
                if model_name in top_models_set:
                    pred_visibility[model_name] = show_checked
                else:
                    # Non-top-3 models by RMSE never show predictions
                    pred_visibility[model_name] = False

            st.session_state["test_visibility"] = test_visibility
            st.session_state["pred_visibility"] = pred_visibility
            
            # Selected models for test data display
            test_selected_models = {name for name, v in test_visibility.items() if v}
            # Selected models for prediction display
            pred_selected_models = {name for name, v in pred_visibility.items() if v}
            
            # When nothing is selected, show no model lines (no automatic fallback)

            # Compute future forecasts once per dataset+future horizon and cache them
            if has_future_predictions:
                try:
                    ser_ds = future_rows["ds"]
                    if pd.api.types.is_datetime64_any_dtype(ser_ds):
                        key_vals = ser_ds.dt.strftime('%Y-%m-%d').tolist()
                    else:
                        key_vals = pd.to_datetime(ser_ds, errors="coerce").dt.strftime('%Y-%m-%d').tolist()
                    fr_ds_key = ",".join(key_vals)
                except Exception:
                    fr_ds_key = str(len(future_rows))
                future_key = f"{_df_fingerprint(series)}|future={fr_ds_key}"
                future_cache = st.session_state.get("future_cache", {})

                # Build a map of predictions per model if not cached
                if future_key not in future_cache:
                    try:
                        with st.spinner("Forecasting future values..."):
                            # Build mapping for exact and base display name -> estimator
                            reg_estimators = list(get_fast_estimators())
                            name_to_est_full = {name: est for name, est in reg_estimators}
                            base_to_est = {}
                            for disp_name, est in reg_estimators:
                                base = str(disp_name).split(" (")[0]
                                if base not in base_to_est:
                                    base_to_est[base] = est
                            model_to_future = {}
                            model_to_future_time = {}
                            for res in results:
                                model_name = res.get("name")
                                # Restrict future predictions to top 3 models by RMSE only
                                if model_name not in top_models_set:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    model_to_future_time[model_name] = None
                                    continue
                                # Handle +Trend models specially
                                is_trend_model = "+Trend" in model_name
                                base_model_name = model_name.replace("+Trend", "") if is_trend_model else model_name
                                # Prefer exact match first, then base name, then prefix fallback
                                est = name_to_est_full.get(base_model_name) or base_to_est.get(base_model_name)
                                if est is None:
                                    # Try prefix match against registry keys
                                    for disp_name, cand in reg_estimators:
                                        if str(disp_name).startswith(base_model_name):
                                            est = cand
                                            break
                                if est is None:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    model_to_future_time[model_name] = None
                                    continue
                                params = res.get("estimator_params", {})
                                try:
                                    if params:
                                        est.set_params(**params)
                                except Exception:
                                    pass
                                try:
                                    t_start_future = time.time()
                                    if is_trend_model:
                                        # For trend models, we need to handle detrending manually
                                        # since train_on_known_and_forecast_missing doesn't support it
                                        from trend import fit_trend
                                        
                                        # Fit trend on known data
                                        known_data = series.dropna(subset=["y"]).copy()
                                        trend_model = fit_trend(known_data)
                                        
                                        # Create detrended series for training
                                        detrended_series = series.copy()
                                        trend_fitted = trend_model.fitted(detrended_series["ds"])
                                        detrended_series.loc[detrended_series["y"].notna(), "y"] = (
                                            detrended_series.loc[detrended_series["y"].notna(), "y"] - 
                                            trend_fitted[detrended_series["y"].notna()].values
                                        )
                                        
                                        # Train on detrended data
                                        res_future_df = train_on_known_and_forecast_missing(
                                            detrended_series, est, future_rows=future_rows
                                        )
                                        
                                        # Add trend back to predictions
                                        if not res_future_df.empty:
                                            future_trend = trend_model.extrapolate(res_future_df["ds"])
                                            res_future_df["yhat"] = res_future_df["yhat"] + future_trend.values
                                    else:
                                        # Regular model, no detrending
                                        res_future_df = train_on_known_and_forecast_missing(
                                            series, est, future_rows=future_rows
                                        )
                                    t_end_future = time.time()
                                    if res_future_df.empty:
                                        # provide at least a flat continuation to avoid blank UI
                                        res_future_df = pd.DataFrame({
                                            "ds": future_rows["ds"],
                                            "yhat": [series["y"].dropna().iloc[-1]] * len(future_rows)
                                        })
                                    model_to_future_time[model_name] = float(max(0.0, t_end_future - t_start_future))
                                except Exception:
                                    res_future_df = pd.DataFrame({
                                        "ds": future_rows["ds"],
                                        "yhat": [series["y"].dropna().iloc[-1] if series["y"].notna().any() else 0.0] * len(future_rows)
                                    })
                                    model_to_future_time[model_name] = None
                                model_to_future[model_name] = res_future_df
                            future_cache[future_key] = {"preds": model_to_future, "times": model_to_future_time}
                            st.session_state["future_cache"] = future_cache
                    except Exception as e:
                        st.error(f"Forecasting failed: {e}")

                # Attach cached future forecasts to results
                cache_entry = future_cache.get(future_key, {})
                if isinstance(cache_entry, dict) and ("preds" in cache_entry or "times" in cache_entry):
                    model_to_future = cache_entry.get("preds", {})
                    model_to_future_time = cache_entry.get("times", {})
                else:
                    # Backward-compat: older cache stored a flat map of model->future_df
                    model_to_future = cache_entry if isinstance(cache_entry, dict) else {}
                    model_to_future_time = {k: None for k in model_to_future.keys()}
                for res in results:
                    model_name = res.get("name")
                    # Ensure only top unique models have future predictions; others remain empty
                    if model_name in top_models_set:
                        res["future_df"] = model_to_future.get(model_name, pd.DataFrame({"ds": [], "yhat": []}))
                        res["predict_future_time_s"] = model_to_future_time.get(model_name, None)
                    else:
                        res["future_df"] = pd.DataFrame({"ds": [], "yhat": []})
                        res["predict_future_time_s"] = None

                # Set horizon from any model with predictions
                for res in results:
                    if isinstance(res.get("future_df"), pd.DataFrame) and not res["future_df"].empty:
                        future_horizon = len(res["future_df"]) or 0
                        break

        # Compute simple trend over the whole known series
        trend_df_plot = None
        try:
            known = series.dropna(subset=["y"]).copy()
            if not known.empty:
                tm = fit_trend(known)
                trend_vals = tm.fitted(series["ds"]) if "ds" in series.columns else None
                if trend_vals is not None:
                    trend_df_plot = pd.DataFrame({"ds": series["ds"], "trend": trend_vals})
        except Exception:
            trend_df_plot = None

        # Show trend description in the left checklist area
        try:
            if _trend_desc is not None:
                with progress_container:
                    icon = "⚠️" if str(_trend_desc).strip().lower() == "no trend detected" else "✅"
                    st.markdown(
                        f"<div style='margin:2px 0; line-height:1.2'>{icon} Trend: {_trend_desc}</div>",
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

        # Create and display plot using utility function (rendered above the table)
        fig = create_forecast_plot(
            series=series, results=results, future_df=None,
            show_only_test=show_only_test, split_ds=split_ds,
            hide_non_chosen_models=False, best_name=best_name,
            visible_test_models=test_selected_models,
            visible_pred_models=pred_selected_models,
            trend_series=trend_df_plot,
        )
        with plot_container:
            st.pyplot(fig)
        # Training data preview removed per request

        # Results table rendered below with per-model checkboxes

        # Checklist is already rendered progressively during validation
        # No need to render again here

        # Offer CSV download with predicted target (after checklist, before settings)
        # Only show download when we have future predictions
        if has_future_predictions:
            try:
                best_future_df = None
                if isinstance(results, list) and len(results) > 0:
                    best_res = sorted(results, key=lambda r: r.get("rmse", float("inf")))[0]
                    best_future_df = best_res.get("future_df")
                if isinstance(best_future_df, pd.DataFrame) and not best_future_df.empty:
                    time_col = load_meta.get("original_time_col", raw_df.columns[0] if not raw_df.empty else "date")
                    target_col = load_meta.get("original_target_col", raw_df.columns[-1] if not raw_df.empty else "target")
                    pred_col = f"predicted_{str(target_col)}"
                    # Map ds -> yhat for future rows
                    try:
                        pred_map = dict(zip(pd.to_datetime(best_future_df["ds"], errors="coerce").dt.normalize(), best_future_df["yhat"]))
                    except Exception:
                        pred_map = {}
                    download_df = raw_df.copy()
                    try:
                        # Use detected date format if available to avoid re-parsing ambiguities
                        detected_fmt = file_info.get("detected_date_format") if isinstance(file_info, dict) else None
                        if detected_fmt:
                            ds_norm = pd.to_datetime(download_df[time_col], errors="coerce", format=str(detected_fmt)).dt.normalize()
                        else:
                            ds_norm = pd.to_datetime(download_df[time_col], errors="coerce", dayfirst=True).dt.normalize()
                        download_df[pred_col] = ds_norm.map(pred_map)
                        # Ensure date column is formatted as dd/mm/yyyy for CSV output
                        try:
                            download_df[time_col] = ds_norm.dt.strftime("%d/%m/%Y")
                        except Exception:
                            pass
                    except Exception:
                        try:
                            ds_norm = pd.to_datetime(download_df[time_col], errors="coerce").dt.normalize()
                            download_df[pred_col] = ds_norm.map(pred_map)
                            # Ensure date column is formatted as dd/mm/yyyy for CSV output
                            try:
                                download_df[time_col] = ds_norm.dt.strftime("%d/%m/%Y")
                            except Exception:
                                pass
                        except Exception:
                            download_df[pred_col] = None
                            # Best-effort formatting of date column even if prediction mapping failed
                            try:
                                _tmp_dt = pd.to_datetime(download_df[time_col], errors="coerce", dayfirst=True)
                                download_df[time_col] = _tmp_dt.dt.strftime("%d/%m/%Y")
                            except Exception:
                                pass
                    out_name = f"{Path(data_source_name).stem}_with_predictions.csv"
                    st.sidebar.markdown("<div style='font-weight:600; margin:6px 0 6px 0; text-align:center'>Result</div>", unsafe_allow_html=True)
                    st.sidebar.download_button(
                        label="Download CSV with predictions",
                        data=download_df.to_csv(index=False).encode("utf-8"),
                        file_name=out_name,
                        mime="text/csv",
                    )
            except Exception:
                pass
        else:
            # When no future predictions, append a warning in the checklist
            with progress_container:
                st.markdown(
                    "<div style='margin:2px 0; line-height:1.2'>⚠️ No future predictions available. The model comparison table shows training performance on your historical data.</div>",
                    unsafe_allow_html=True,
                )

        # Sidebar display toggles (after results)
        with st.sidebar:
            st.markdown("<div style='font-weight:600; margin:6px 0 6px 0; text-align:center'>Settings</div>", unsafe_allow_html=True)
            st.checkbox(
                "Show full history",
                key="show_full_history",
                value=st.session_state.get("show_full_history", False)
            )
            st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)


except Exception as e:
    # Checklist is already shown progressively, just show the error
    try:
        st.error(f"Unexpected error: {e}")
    except Exception:
        pass
