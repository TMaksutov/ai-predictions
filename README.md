# Time Series Forecasting Benchmark

A Streamlit web application for benchmarking Prophet time series forecasting on sample datasets using an optimized one-page layout.

## Overview

This application provides:
- **Benchmark results** for Prophet model only
- **RMSE evaluation** on the last 20% of each dataset (holdout test set)
- **Interactive visualization** of forecasts for selected datasets
- **10 diverse sample datasets** covering different time series patterns
- **Optimized layout** with table on left, graph on right, all content fitting on one page
- **Click-to-select**: Pick a dataset by clicking its row in the Benchmark Results table

## Layout Features

- **Two-Column Design**: Benchmark results table on the left, forecast visualization on the right
- **No Scrolling Required**: All content fits on one page for better user experience
- **Compact Header**: Streamlined title and navigation for maximum content space
- **Responsive Metrics**: Key statistics displayed in organized metric cards

## Features

### Benchmark Model
- **Prophet**: Automatic seasonality detection with configurable priors

### Sample Datasets
The application includes 10 synthetic datasets with various characteristics:
- Linear trends with noise
- Seasonal patterns (weekly, monthly, yearly)
- Exponential growth and decay
- Cyclical patterns
- Step changes and volatility patterns

### Evaluation Methodology
- **Training**: Uses the first 80% of each dataset
- **Testing**: Evaluates on the last 20% of each dataset
- **Metric**: Root Mean Square Error (RMSE)
- **Metric Only**: RMSE results for Prophet

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The application will:
1. Display a benchmark table with Prophet RMSE across all 10 datasets (left column)
2. Select a dataset by clicking its row in the Benchmark Results table (or via the selector)
3. Show the Prophet forecast plot with confidence intervals in a compact layout

## Dependencies

- `streamlit`: Web application framework (>=1.29.0)
- `pandas`: Data manipulation and analysis (>=2.1.0)
- `numpy`: Numerical computing (>=1.24.0)
- `matplotlib`: Plotting and visualization (>=3.7.0)
- `prophet`: Forecasting tool (>=1.1.4)
- `plotly`: Additional plotting capabilities (>=5.15.0)

### Updating Dependencies

To update dependencies in the future:

1. **Check for updates**:
   ```bash
   pip list --outdated
   ```

2. **Update specific packages**:
   ```bash
   pip install --upgrade package_name
   ```

3. **Update requirements.txt**:
   ```bash
   pip freeze > requirements.txt
   ```

## AI Agents Instructions

This project includes AI agent instructions for systematic maintenance and updates. See `agents.md` for detailed guidance.

### For AI Agents Working on This Project

1. **Read First**: Always consult `agents.md` for current instructions and guidelines
2. **Layout Constraints**: Maintain two-column layout with no scrolling required
3. **Code Standards**: Follow established patterns for caching, layout, and performance
4. **Update Protocol**: Use the instruction check and update procedures in `agents.md`

### Updating Agents in the Future

When new instructions are required for AI agents:

1. **Modify `agents.md`**: Update the instruction file with new requirements
2. **Follow Template**: Use the update template provided in `agents.md`
3. **Document Changes**: Add entries to the update history section
4. **Validate**: Test that new instructions work correctly
5. **Version Control**: Commit changes with descriptive messages

Example update process:
```bash
# Edit the agents instruction file
nano agents.md

# Test the changes work
streamlit run streamlit_app.py

# Commit the updated instructions
git add agents.md
git commit -m "Update agent instructions: [description]"
```

## Dataset Information

All datasets are synthetically generated with known patterns for benchmarking purposes. Each dataset contains:
- **Date column** (`ds`): Daily timestamps
- **Value column** (`y`): Time series values
- **Length**: 150-500 data points per dataset

## Model Details

### Prophet Model
- Automatic seasonality detection (daily, weekly, yearly)
- Configurable parameters for changepoint and seasonality prior scales
- Built-in holiday effects and trend analysis

## Performance Notes

- Results are cached for faster subsequent runs
- Prophet may take longer to train but often provides better accuracy
- The application processes all 10 datasets for benchmarking

## Related Work

This benchmarking application is inspired by the TSForecasting repository. For more comprehensive forecasting research and datasets, see:
- **TSForecasting Repository**: [TSForecasting_README.md](TSForecasting_README.md)
- **AI Agent Instructions**: [agents.md](agents.md)
- **Monash Time Series Forecasting Archive**: https://forecastingdata.org/

## Contributing

Feel free to contribute by:
- Adding new forecasting models
- Including additional evaluation metrics
- Expanding the dataset collection
- Improving the user interface
- Updating AI agent instructions for better automation

## License

This project is open source and available under the MIT License.