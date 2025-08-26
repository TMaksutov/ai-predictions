from pathlib import Path

import pandas as pd
import streamlit as st
from modeling import get_fast_estimators
from modeling import train_on_known_and_forecast_missing, UnifiedTimeSeriesTrainer
from features import build_features as _build_features
from data_io import load_data_with_checklist, validate_data_with_checklist
from data_utils import load_default_dataset, prepare_series_from_dataframe, get_future_rows
from plot_utils import create_forecast_plot, create_results_table
import io
import uuid
import json
import os
from urllib import request

st.set_page_config(
    page_title="Daily Data Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        for idx, ch in enumerate(model_name):
            if ch.isdigit():
                return model_name[:idx] or model_name
        return model_name
    except Exception:
        return str(model_name)


def _pick_top_unique_models(results: list, max_models: int = 3) -> list:
    """Pick up to N best models with unique base types by RMSE."""
    chosen = []
    seen_types = set()
    try:
        for r in sorted(results, key=lambda x: x.get("rmse", float("inf"))):
            name = r.get("name")
            if not name:
                continue
            base = _base_model_type(name)
            if base in seen_types:
                continue
            seen_types.add(base)
            chosen.append(name)
            if len(chosen) >= max_models:
                break
    except Exception:
        pass
    return chosen


 # -----------------------------
 # UI
 # -----------------------------
# Data loading - simplified using utility functions
uploaded = st.sidebar.file_uploader(
    "Upload a CSV (date column first, target last)",
    type=["csv"],
    accept_multiple_files=False,
    help="If no file is uploaded, the default dataset 'sample.csv' will be used."
)

# Privacy note ( upload counts only; no personal data)
st.sidebar.caption("This app collects  upload counts only. No personal data.")

 

# Load data using simplified logic
if uploaded is not None:
    data_source_name = getattr(uploaded, "name", "uploaded.csv")
    raw_df, file_info = load_data_with_checklist(uploaded)
    # Count a single upload event per user session; no PII sent
    try:
        if not st.session_state.get("ga_uploaded_once", False):
            _ga_track("file_uploaded")
            st.session_state["ga_uploaded_once"] = True
    except Exception:
        pass
else:
    data_source_name = SAMPLE_PATH.name
    raw_df, file_info = load_default_dataset(SAMPLE_PATH)

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

if data_source_name:
    try:
        st.sidebar.caption(f"Dataset: {data_source_name}")
    except Exception:
        pass

# Sidebar checklist
progress_container = st.sidebar.container()
with progress_container:
    st.markdown("<div style='font-weight:600; margin:0 0 6px 0; text-align:center'>Checklist</div>", unsafe_allow_html=True)

def _render_checklist(items):
    with progress_container:
        st.markdown(f"<div style='font-weight:600; margin:6px 0 3px 0'>Validation Checklist</div>", unsafe_allow_html=True)
        for status, text in items:
            icon = "✅" if status == "ok" else ("⚠️" if status == "warn" else "❌")
            st.markdown(f"<div style='margin:2px 0; line-height:1.2'>{icon} {text}</div>", unsafe_allow_html=True)

# Persisted UI defaults for display controls
for _key, _default in [
    ("show_only_test", True),
    ("hide_features_table", True),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# New user-facing toggles default to sensible values
for _key, _default in [
    ("show_full_history", False),
    ("show_training_data_preview", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Keep legacy flag in sync
st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)

# Read current control values from session state (widgets will render under the graph)
show_only_test = st.session_state.get("show_only_test", True)
hide_features_table = st.session_state.get("hide_features_table", True)

# Build checklist early and gate predictions/features if any non-OK in critical sections
checklist_items, is_valid = validate_data_with_checklist(raw_df, file_info)

if not is_valid:
    _render_checklist(checklist_items)
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

        future_rows, last_observed_idx = get_future_rows(series, feature_cols)
        if future_rows.empty:
            st.error("No future rows detected. Append future dates (and features if present) with empty target at the end.")
            st.stop()

        # Benchmark only on known rows
        series_for_benchmark = series.iloc[: last_observed_idx + 1].dropna(subset=["y"]).copy()
        # Import config for default values
        try:
            from config import DEFAULT_TEST_FRACTION
        except ImportError:
            DEFAULT_TEST_FRACTION = 0.2

        # Cache benchmark results so UI toggles do not retrigger training
        bench_key = f"{_df_fingerprint(series_for_benchmark)}|test_frac={DEFAULT_TEST_FRACTION}"
        bench_cache = st.session_state.get("bench_cache", {})
        if bench_key in bench_cache:
            results = bench_cache[bench_key]
        else:
            try:
                with st.spinner("Training and evaluating models..."):
                    results = run_benchmark(series_for_benchmark, test_fraction=DEFAULT_TEST_FRACTION)
                bench_cache[bench_key] = results
                st.session_state["bench_cache"] = bench_cache
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
                        {"name": n, "rmse": float("inf"), "mape": None,
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
        # Compute top unique models (max 3, unique base type)
        top_unique_models = _pick_top_unique_models(results, max_models=3)
        top_unique_set = set(top_unique_models)
        if best_name is not None:
            # Initialize or reconcile visibility state
            available_models = [r["name"] for r in results]
            visibility = st.session_state.get("model_visibility")
            prev_key = st.session_state.get("visibility_key")
            if (not isinstance(visibility, dict)) or (set(visibility.keys()) != set(available_models)):
                # Model set changed: reset defaults to top unique models
                visibility = {name: (name in top_unique_set) for name in available_models}
                st.session_state["model_visibility"] = visibility
                st.session_state["visibility_key"] = bench_key
            elif prev_key != bench_key:
                # Dataset changed: reset defaults to top unique models for the new dataset
                visibility = {name: (name in top_unique_set) for name in available_models}
                st.session_state["model_visibility"] = visibility
                st.session_state["visibility_key"] = bench_key

            # Apply any pending edits from the editor state so toggles reflect immediately on rerun
            try:
                editor_state = st.session_state.get("model_table_editor", None)
                if isinstance(editor_state, dict) and isinstance(editor_state.get("edited_rows"), dict):
                    index_to_model = [r["name"] for r in results]
                    for idx, changes in editor_state["edited_rows"].items():
                        try:
                            if 0 <= idx < len(index_to_model):
                                model_name = index_to_model[idx]
                                if model_name in visibility and "Show" in changes:
                                    visibility[model_name] = bool(changes["Show"])
                        except Exception:
                            pass
                    st.session_state["model_visibility"] = visibility
            except Exception:
                pass

            # Table will render below the plot

            # Create a compact benchmark table using st.data_editor
            def _fmt(v):
                try:
                    import math
                    if v is None:
                        return ""
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
                default_checked = bool(visibility.get(m, m in top_unique_set))
                # Convert mape (relative) to percent for computing accuracy only; not displayed
                mape_rel = r.get('mape', None)
                mape_pct = (mape_rel * 100.0) if isinstance(mape_rel, (int, float)) and pd.notna(mape_rel) else None
                acc_pct = (100.0 - mape_pct) if isinstance(mape_pct, float) and pd.notna(mape_pct) else None
                table_data.append({
                    "Show": default_checked,
                    "Model": m,
                    f"{METRIC_NAME}": _fmt(r.get('rmse', None)),
                    "Accuracy (%)": _fmt(acc_pct),
                    "Train (s)": _fmt(r.get('train_time_s', None)),
                    "Predict (s)": _fmt(r.get('predict_time_s', None))
                })

            results_df = pd.DataFrame(table_data)

            with table_container:
                edited_df = st.data_editor(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Show": st.column_config.CheckboxColumn(
                            "Show",
                            help="Select models to display in the plot",
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
                            help="Approximate accuracy derived from percentage error. Info only.",
                            disabled=True,
                        ),
                        "Train (s)": st.column_config.TextColumn(
                            "Train (s)",
                            disabled=True,
                        ),
                        "Predict (s)": st.column_config.TextColumn(
                            "Predict (s)",
                            disabled=True,
                        ),
                    },
                    key="model_table_editor"
                )

            

            for _, row in edited_df.iterrows():
                model_name = row["Model"]
                if model_name in visibility:
                    visibility[model_name] = bool(row["Show"])

            st.session_state["model_visibility"] = visibility
            selected_models = {name for name, v in visibility.items() if v}
            if not selected_models and best_name is not None:
                selected_models = {best_name}

            # Compute future forecasts once per dataset+future horizon and cache them
            if not future_rows.empty:
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
                            name_to_est = {name: est for name, est in get_fast_estimators()}
                            model_to_future = {}
                            for res in results:
                                model_name = res.get("name")
                                # Restrict future predictions to top unique models only
                                if model_name not in top_unique_set:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    continue
                                est = name_to_est.get(model_name)
                                if est is None:
                                    model_to_future[model_name] = pd.DataFrame({"ds": [], "yhat": []})
                                    continue
                                params = res.get("estimator_params", {})
                                try:
                                    if params:
                                        est.set_params(**params)
                                except Exception:
                                    pass
                                try:
                                    res_future_df = train_on_known_and_forecast_missing(
                                        series, est, future_rows=future_rows
                                    )
                                    if res_future_df.empty:
                                        # provide at least a flat continuation to avoid blank UI
                                        res_future_df = pd.DataFrame({
                                            "ds": future_rows["ds"],
                                            "yhat": [series["y"].dropna().iloc[-1]] * len(future_rows)
                                        })
                                except Exception:
                                    res_future_df = pd.DataFrame({
                                        "ds": future_rows["ds"],
                                        "yhat": [series["y"].dropna().iloc[-1] if series["y"].notna().any() else 0.0] * len(future_rows)
                                    })
                                model_to_future[model_name] = res_future_df
                            future_cache[future_key] = model_to_future
                            st.session_state["future_cache"] = future_cache
                    except Exception as e:
                        st.error(f"Forecasting failed: {e}")

                # Attach cached future forecasts to results
                model_to_future = future_cache.get(future_key, {})
                for res in results:
                    model_name = res.get("name")
                    # Ensure only top unique models have future predictions; others remain empty
                    if model_name in top_unique_set:
                        res["future_df"] = model_to_future.get(model_name, pd.DataFrame({"ds": [], "yhat": []}))
                    else:
                        res["future_df"] = pd.DataFrame({"ds": [], "yhat": []})

                # Set horizon from any model with predictions
                for res in results:
                    if isinstance(res.get("future_df"), pd.DataFrame) and not res["future_df"].empty:
                        future_horizon = len(res["future_df"]) or 0
                        break

        # Create and display plot using utility function (rendered above the table)
        fig = create_forecast_plot(
            series=series, results=results, future_df=None,
            show_only_test=show_only_test, split_ds=split_ds,
            hide_non_chosen_models=False, best_name=best_name,
            visible_models=selected_models,
        )
        with plot_container:
            st.pyplot(fig)
        # Optional: show the dataframe used for training/prediction (first 5 rows)
        if not hide_features_table:
            try:
                features_df, feature_cols = _build_features(series)
                st.markdown("#### Training/prediction data (first 5 rows)")
                st.dataframe(features_df.head(5), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not build features preview: {e}")

        # Results table rendered below with per-model checkboxes

        # Render checklist in sidebar grouped by module
        try:
            _render_checklist(checklist_items)
        except Exception:
            pass

        # Offer CSV download with predicted target (after checklist, before settings)
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
                except Exception:
                    try:
                        ds_norm = pd.to_datetime(download_df[time_col], errors="coerce").dt.normalize()
                        download_df[pred_col] = ds_norm.map(pred_map)
                    except Exception:
                        download_df[pred_col] = None
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

        # Sidebar display toggles (placed after checklist)
        with st.sidebar:
            st.markdown("<div style='font-weight:600; margin:6px 0 6px 0; text-align:center'>Settings</div>", unsafe_allow_html=True)
            st.checkbox(
                "Show full history",
                key="show_full_history",
                value=st.session_state.get("show_full_history", False)
            )
            st.session_state["show_only_test"] = not st.session_state.get("show_full_history", False)

            st.checkbox(
                "Show training data preview",
                key="show_training_data_preview",
                value=st.session_state.get("show_training_data_preview", False)
            )
            st.session_state["hide_features_table"] = not st.session_state.get("show_training_data_preview", False)

except Exception as e:
    # Show checklist and the error so the UI does not appear frozen
    _render_checklist(checklist_items)
    try:
        st.error(f"Unexpected error: {e}")
    except Exception:
        pass
