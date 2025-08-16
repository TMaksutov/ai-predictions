from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import importlib
from models.prophet_model import forecast_and_nrmse as prophet_forecast_and_nrmse


st.set_page_config(page_title="TS Forecasting Benchmark", layout="wide")

# Compact header
st.markdown("### Time Series Forecasting Benchmark")
st.caption("NRMSE on 10 sample datasets")


def _generate_sample_datasets() -> Dict[str, pd.DataFrame]:
	"""Generate 10 diverse sample time series datasets for benchmarking."""
	datasets = {}
	np.random.seed(42)  # For reproducible results
	
	# Dataset 1: Linear trend with noise
	dates = pd.date_range('2020-01-01', periods=200, freq='D')
	trend = np.linspace(100, 200, 200)
	noise = np.random.normal(0, 5, 200)
	datasets['Linear_Trend'] = pd.DataFrame({
		'ds': dates,
		'y': trend + noise
	})
	
	# Dataset 2: Seasonal pattern (weekly)
	dates = pd.date_range('2020-01-01', periods=300, freq='D')
	seasonal = 50 + 20 * np.sin(2 * np.pi * np.arange(300) / 7)  # Weekly pattern
	noise = np.random.normal(0, 3, 300)
	datasets['Weekly_Seasonal'] = pd.DataFrame({
		'ds': dates,
		'y': seasonal + noise
	})
	
	# Dataset 3: Exponential growth
	dates = pd.date_range('2020-01-01', periods=150, freq='D')
	growth = 10 * np.exp(0.01 * np.arange(150))
	noise = np.random.normal(0, growth * 0.05)  # Proportional noise
	datasets['Exponential_Growth'] = pd.DataFrame({
		'ds': dates,
		'y': growth + noise
	})
	
	# Dataset 4: Multiple seasonality (daily + weekly)
	dates = pd.date_range('2020-01-01', periods=400, freq='H')
	daily = 10 * np.sin(2 * np.pi * np.arange(400) / 24)  # Daily pattern
	weekly = 5 * np.sin(2 * np.pi * np.arange(400) / (24*7))  # Weekly pattern
	base = 100
	noise = np.random.normal(0, 2, 400)
	datasets['Hourly_Multi_Seasonal'] = pd.DataFrame({
		'ds': dates,
		'y': base + daily + weekly + noise
	})
	
	# Dataset 5: Step change
	dates = pd.date_range('2020-01-01', periods=250, freq='D')
	values = np.ones(250) * 50
	values[125:] += 30  # Step change halfway
	noise = np.random.normal(0, 4, 250)
	datasets['Step_Change'] = pd.DataFrame({
		'ds': dates,
		'y': values + noise
	})
	
	# Dataset 6: Cyclical pattern (annual-like)
	dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
	annual = 75 + 25 * np.sin(2 * np.pi * np.arange(365*2) / 365)
	noise = np.random.normal(0, 3, 365*2)
	datasets['Annual_Cycle'] = pd.DataFrame({
		'ds': dates,
		'y': annual + noise
	})
	
	# Dataset 7: Random walk
	dates = pd.date_range('2020-01-01', periods=200, freq='D')
	random_walk = np.cumsum(np.random.normal(0, 2, 200)) + 100
	datasets['Random_Walk'] = pd.DataFrame({
		'ds': dates,
		'y': random_walk
	})
	
	# Dataset 8: Polynomial trend
	dates = pd.date_range('2020-01-01', periods=180, freq='D')
	x = np.arange(180) / 180
	polynomial = 50 + 30*x + 20*x**2 - 10*x**3
	noise = np.random.normal(0, 4, 180)
	datasets['Polynomial_Trend'] = pd.DataFrame({
		'ds': dates,
		'y': polynomial + noise
	})
	
	# Dataset 9: Damped oscillation
	dates = pd.date_range('2020-01-01', periods=220, freq='D')
	t = np.arange(220)
	damped = 100 + 50 * np.exp(-t/100) * np.sin(2 * np.pi * t / 30)
	noise = np.random.normal(0, 3, 220)
	datasets['Damped_Oscillation'] = pd.DataFrame({
		'ds': dates,
		'y': damped + noise
	})
	
	# Dataset 10: Mixed patterns
	dates = pd.date_range('2020-01-01', periods=300, freq='D')
	trend = 0.1 * np.arange(300)
	seasonal = 15 * np.sin(2 * np.pi * np.arange(300) / 50)  # Custom period
	weekly = 5 * np.sin(2 * np.pi * np.arange(300) / 7)
	base = 80
	noise = np.random.normal(0, 4, 300)
	datasets['Mixed_Patterns'] = pd.DataFrame({
		'ds': dates,
		'y': base + trend + seasonal + weekly + noise
	})
	
	return datasets


def _get_dataset_names() -> List[str]:
	"""Get list of available dataset names."""
	return [
		'Linear_Trend', 'Weekly_Seasonal', 'Exponential_Growth', 
		'Hourly_Multi_Seasonal', 'Step_Change', 'Annual_Cycle',
		'Random_Walk', 'Polynomial_Trend', 'Damped_Oscillation', 'Mixed_Patterns'
	]


@st.cache_data(show_spinner=False)
def _load_dataset(name: str) -> Tuple[pd.DataFrame, dict]:
	"""Load a specific dataset by name."""
	datasets = _generate_sample_datasets()
	if name not in datasets:
		raise ValueError(f"Dataset {name} not found")
	
	data = datasets[name].copy()
	metadata = {"name": name, "description": f"Generated sample dataset: {name}"}
	return data, metadata


def _prepare_single_series(df: pd.DataFrame) -> pd.DataFrame:
	"""Prepare dataset for forecasting."""
	# Already in correct format (ds, y)
	sub = df[["ds", "y"]].copy()
	
	# Ensure proper types
	sub["ds"] = pd.to_datetime(sub["ds"])
	sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
	sub = sub.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
	
	# Deduplicate by timestamp (shouldn't be needed for generated data)
	sub = sub.groupby("ds", as_index=False).agg({"y": "mean"})
	
	return sub


@st.cache_data(show_spinner=False)
def _compute_benchmark(dataset_names: Tuple[str, ...]) -> pd.DataFrame:
	"""Compute benchmark results for all datasets across all models."""
	results = []
	for name in dataset_names:
		try:
			raw_df, _ = _load_dataset(name)
			series_df = _prepare_single_series(raw_df)
			
			row = {"Dataset": name}
			# Prophet
			try:
				p_nrmse, _, _ = prophet_forecast_and_nrmse(series_df, test_fraction=0.2, optimize_params_flag=False)
				row["Prophet NRMSE"] = f"{p_nrmse:.4f}"
			except Exception:
				row["Prophet NRMSE"] = "Error"
			# ARIMA / SARIMA
			try:
				arima_mod = importlib.import_module('models.arima_model')
				a_nrmse, _, _ = arima_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["ARIMA NRMSE"] = f"{a_nrmse:.4f}"
			except Exception:
				row["ARIMA NRMSE"] = "Error"
			# Holt-Winters
			try:
				hw_mod = importlib.import_module('models.holtwinters_model')
				h_nrmse, _, _ = hw_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["Holt-Winters NRMSE"] = f"{h_nrmse:.4f}"
			except Exception:
				row["Holt-Winters NRMSE"] = "Error"
			# LightGBM
			try:
				lgbm_mod = importlib.import_module('models.lightgbm_model')
				l_nrmse, _, _ = lgbm_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["LightGBM NRMSE"] = f"{l_nrmse:.4f}"
			except Exception:
				row["LightGBM NRMSE"] = "Error"
			# XGBoost
			try:
				xgb_mod = importlib.import_module('models.xgboost_model')
				x_nrmse, _, _ = xgb_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["XGBoost NRMSE"] = f"{x_nrmse:.4f}"
			except Exception:
				row["XGBoost NRMSE"] = "Error"
			# Random Forest
			try:
				rf_mod = importlib.import_module('models.random_forest_model')
				r_nrmse, _, _ = rf_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["RandomForest NRMSE"] = f"{r_nrmse:.4f}"
			except Exception:
				row["RandomForest NRMSE"] = "Error"
			# AutoTS
			try:
				autots_mod = importlib.import_module('models.autots_model')
				at_nrmse, _, _ = autots_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["AutoTS NRMSE"] = f"{at_nrmse:.4f}"
			except Exception:
				row["AutoTS NRMSE"] = "Error"
			# Darts
			try:
				darts_mod = importlib.import_module('models.darts_models')
				d_nrmse, _, _ = darts_mod.forecast_and_nrmse(series_df, test_fraction=0.2)
				row["Darts NRMSE"] = f"{d_nrmse:.4f}"
			except Exception:
				row["Darts NRMSE"] = "Error"
			
			results.append(row)
		except Exception:
			results.append({
				"Dataset": name,
				"Prophet NRMSE": "Error",
				"ARIMA NRMSE": "Error",
				"Holt-Winters NRMSE": "Error",
				"LightGBM NRMSE": "Error",
				"XGBoost NRMSE": "Error",
				"RandomForest NRMSE": "Error",
				"AutoTS NRMSE": "Error",
				"Darts NRMSE": "Error",
			})
	return pd.DataFrame(results)


# Load dataset names
dataset_names = _get_dataset_names()

# Benchmark section (full width)
st.markdown("#### Benchmark Results")
with st.spinner("Computing benchmark..."):
	bench_df = _compute_benchmark(tuple(dataset_names))

# Click-to-select dataset via checkbox column
if "selected_dataset" not in st.session_state:
	st.session_state["selected_dataset"] = dataset_names[0]

bench_df = bench_df.copy()
bench_df.insert(0, "Select", bench_df["Dataset"] == st.session_state["selected_dataset"])
edited_df = st.data_editor(
	bench_df,
	use_container_width=True,
	height=420,
	hide_index=True,
	column_config={
		"Select": st.column_config.CheckboxColumn("Select", help="Click to visualize this dataset", default=False),
		"Dataset": st.column_config.TextColumn("Dataset", disabled=True),
		"Prophet NRMSE": st.column_config.TextColumn("Prophet NRMSE", disabled=True),
		"ARIMA NRMSE": st.column_config.TextColumn("ARIMA NRMSE", disabled=True),
		"Holt-Winters NRMSE": st.column_config.TextColumn("Holt-Winters NRMSE", disabled=True),
		"LightGBM NRMSE": st.column_config.TextColumn("LightGBM NRMSE", disabled=True),
		"XGBoost NRMSE": st.column_config.TextColumn("XGBoost NRMSE", disabled=True),
		"RandomForest NRMSE": st.column_config.TextColumn("RandomForest NRMSE", disabled=True),
		"AutoTS NRMSE": st.column_config.TextColumn("AutoTS NRMSE", disabled=True),
		"Darts NRMSE": st.column_config.TextColumn("Darts NRMSE", disabled=True),
	},
)
selected_rows = edited_df[edited_df["Select"] == True]
if len(selected_rows) > 0:
	st.session_state["selected_dataset"] = selected_rows["Dataset"].iloc[-1]

# Forecast visualization moved after benchmark
st.markdown("#### Forecast Visualization")
selected = st.session_state.get("selected_dataset")
if selected:
	with st.spinner(f"Generating forecast for {selected}..."):
		raw_df, metadata = _load_dataset(selected)
		series_df = _prepare_single_series(raw_df)
		nrmse, forecast, test_df = prophet_forecast_and_nrmse(series_df, test_fraction=0.2, optimize_params_flag=False)

	# Compact info display
	st.caption(f"**{selected}** | Prophet NRMSE: {nrmse:.4f} | Points: {len(series_df)} | Test: {len(test_df)}")

	import matplotlib.pyplot as plt

	# Create smaller figure to fit layout
	fig, ax = plt.subplots(figsize=(9, 5))
	
	# Plot full actual data
	ax.plot(series_df["ds"], series_df["y"], color="#333333", label="Actual", linewidth=2, alpha=0.8)

	# Plot forecast for test period only
	test_pred = forecast.tail(len(test_df))
	ax.plot(test_pred["ds"], test_pred["yhat"], color="#1f77b4", linestyle="--", linewidth=2, label="Forecast")
	
	# Add confidence interval for test period
	ax.fill_between(test_pred["ds"], test_pred["yhat_lower"], test_pred["yhat_upper"], 
					color="#1f77b4", alpha=0.2, label="Confidence")

	# Split marker
	if len(test_df) > 0:
		ax.axvline(test_df["ds"].iloc[0], color="#888888", linestyle=":", linewidth=1, label="Train/Test")

	ax.set_xlabel("Date", fontsize=10)
	ax.set_ylabel("Value", fontsize=10)
	ax.set_title(f"Forecast: {selected}", fontsize=12)
	ax.legend(fontsize=9)
	ax.grid(alpha=0.3)
	ax.tick_params(labelsize=9)
	fig.tight_layout()

	st.pyplot(fig)
