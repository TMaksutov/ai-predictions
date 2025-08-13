# Simple Time-Series Predictor (Streamlit)
Minimal Streamlit app to upload CSV/XLS/XLSX or delimited text and generate a small baseline forecast safely.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

The app automatically cleans uploaded tables and infers the time interval (day, week, etc.) after you select the date column.

