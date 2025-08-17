from typing import List

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simple TS Benchmark", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


def list_csv_files(directory: Path) -> List[Path]:
	if not directory.exists():
		directory.mkdir(parents=True, exist_ok=True)
	return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def count_rows_in_csv(file_path: Path) -> int:
	try:
		with file_path.open("r", encoding="utf-8") as f:
			return max(0, sum(1 for _ in f) - 1)
	except Exception:
		return 0


def load_series_from_csv(file_path: Path) -> pd.DataFrame:
	"""
	Assumes first column is timestamp and last column is target.
	Renames to (ds, y), coerces types, sorts by ds.
	"""
	df = pd.read_csv(file_path)
	if df.shape[1] < 2:
		raise ValueError("CSV must have at least two columns: time and target")

	first_col = df.columns[0]
	last_col = df.columns[-1]
	sub = df[[first_col, last_col]].copy()
	sub = sub.rename(columns={first_col: "ds", last_col: "y"})
	sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
	sub["y"] = pd.to_numeric(sub["y"], errors="coerce")
	sub = sub.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
	return sub


def forecast_nrmse(series_df: pd.DataFrame, test_fraction: float = 0.2):
	from models.regression_model import forecast_regression
	return forecast_regression(series_df, test_fraction=test_fraction)


st.markdown("### Daily Regression Benchmark")

# Build table of datasets
csv_files = list_csv_files(DATA_DIR)
index_names = [p.name for p in csv_files]
rows_counts = [count_rows_in_csv(p) for p in csv_files]

metric_col_name = "Regression nRMSE"

# Initialize selection state
if "selection_state" not in st.session_state:
	st.session_state.selection_state = {name: False for name in index_names}
else:
	# Ensure keys match current files
	for name in index_names:
		st.session_state.selection_state.setdefault(name, False)
	# Remove stale entries
	st.session_state.selection_state = {k: v for k, v in st.session_state.selection_state.items() if k in index_names}

# Prepare table data
table_df = pd.DataFrame({
	"Select": [st.session_state.selection_state.get(n, False) for n in index_names],
	"# Rows": rows_counts,
	metric_col_name: [None] * len(index_names),
}, index=index_names)

# Interactive editor for single selection
edited_df = st.data_editor(
	table_df,
	use_container_width=True,
	hide_index=False,
	num_rows="fixed",
	column_config={
		"Select": st.column_config.CheckboxColumn(required=False),
		"# Rows": st.column_config.NumberColumn(disabled=True),
		metric_col_name: st.column_config.NumberColumn(format="%.4f", disabled=True),
	},
	key="datasets_editor",
)

# Update session selection from edits
for fname, row in edited_df.iterrows():
	st.session_state.selection_state[fname] = bool(row.get("Select", False))

selected_files = [name for name, is_sel in st.session_state.selection_state.items() if is_sel]

if len(selected_files) == 0:
	st.info("Select one file to run the forecast.")
elif len(selected_files) > 1:
	st.warning("Please select only one file.")
else:
	# Exactly one selected
	selected_name = selected_files[0]
	selected_path = DATA_DIR / selected_name

	try:
		series = load_series_from_csv(selected_path)
		if len(series) < 30:
			st.error("Insufficient data (need at least 30 rows) in the selected file.")
		else:
			nrmse, forecast_df, test_df = forecast_nrmse(series, test_fraction=0.2)

			# Update and display results table with metric for selected row
			updated_table = edited_df.copy()
			updated_table.loc[selected_name, metric_col_name] = float(nrmse)
			st.dataframe(updated_table, use_container_width=True, hide_index=False)

			# Plot actual vs forecast
			import matplotlib.pyplot as plt
			fig, ax = plt.subplots(figsize=(9, 5))
			ax.plot(series["ds"], series["y"], color="#333333", label="Actual", linewidth=2, alpha=0.8)
			ax.plot(forecast_df["ds"], forecast_df["yhat"], color="#1f77b4", linestyle="--", linewidth=2, label="Forecast")
			if len(test_df) > 0:
				ax.axvline(test_df["ds"].iloc[0], color="#888888", linestyle=":", linewidth=1, label="Train/Test")
			ax.set_xlabel("Date")
			ax.set_ylabel("Value")
			ax.set_title(f"{selected_name} — Regression (nRMSE={nrmse:.4f})")
			ax.legend()
			ax.grid(alpha=0.3)
			fig.tight_layout()
			st.pyplot(fig)
	except Exception as e:
		st.error(f"Failed to run forecast: {e}")
