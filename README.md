# Simple Time-Series Predictor (Streamlit)
Minimal Streamlit app to upload CSV/XLS/XLSX or delimited text and generate a small baseline forecast safely. The helper utilities can also auto-detect feature types (dates, numbers, categories, booleans) and train simple regression models such as RandomForest or LightGBM.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

The app automatically cleans uploaded tables and infers the time interval (day, week, etc.) after you select the date column.

Use `train_regression_models(df, target)` to fit models on any tabular data. Columns are parsed for dates, booleans and categories, missing values are handled, and the last 30% of rows are used for evaluation. LightGBM will be used if installed.

