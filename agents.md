# AI Agents - Instructions

## Project Overview
Time Series Forecasting Benchmark app with Prophet NRMSE evaluation on 10 synthetic datasets.

## Key Requirements
- **Layout**: Two-column layout (table left, graph right) fitting on one page
- **Performance**: Use `@st.cache_data` for computations
- **UI**: Interactive dataset selection via table checkboxes

## Code Guidelines
- Layout: `st.columns([1, 1.2])`
- Figure size: `figsize=(8, 5)`
- Data table height: `height=400`
- Headers: Use `####` markdown instead of `st.subheader`

## Project Structure
```
/workspace/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
└── agents.md                 # This file
```

## Current Features
- Prophet model with parameter optimization toggle
- NRMSE (Normalized RMSE) evaluation
- Interactive visualization with dataset selection
- 10 diverse synthetic datasets for testing

## Instructions for Updates
1. Always maintain one-page layout without scrolling
2. Preserve two-column structure
3. Keep performance optimizations (caching)
4. Update this file when making significant changes

---
**Last Updated**: 2025-01-27