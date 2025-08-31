# AI-Predictions: Daily Time Series Forecasting Tool

A lightweight, fast machine learning application for forecasting daily time series data using multiple scikit-learn regressors. This tool is designed for quick benchmarking and forecasting of business metrics, sales data, or any daily time series where you need to predict future values.

## What This Program Does

This application automatically:
1. **Loads and validates** your daily time series data
2. **Engineers features** including lags, moving averages, seasonal patterns, and trends
3. **Trains multiple ML models** simultaneously for comparison
4. **Benchmarks performance** using RMSE on a test split
5. **Generates forecasts** for future dates you specify
6. **Visualizes results** with interactive plots and performance tables

## How It Works: The Strategy

### Core Philosophy
The program follows a **"fast and furious"** approach - it prioritizes speed and practical results over perfect optimization. It's designed for business users who need quick insights, not data scientists doing deep analysis.

### Feature Engineering Strategy
1. **Temporal Features**: Creates lag features (1, 7, 30, 90, 365 days) to capture recent patterns
2. **Moving Averages**: Calculates rolling averages (7, 30, 90 days) to smooth noise
3. **Seasonal Detection**: Automatically detects and creates Fourier features for weekly, monthly, and yearly patterns
4. **Trend Analysis**: Fits linear/quadratic trends to capture long-term direction
5. **Calendar Features**: Day-of-week encoding for weekly patterns

### Model Selection Strategy
The program uses a curated set of **fast, interpretable models**:
- **Linear Models**: Ridge, Lasso, ElasticNet (good for trend + seasonality)
- **Tree Models**: Decision Trees, Gradient Boosting (good for complex patterns)
- **Neighbor Models**: KNN (good for similar historical patterns)
- **Robust Models**: Huber, SVR (good for noisy data)

### Training Strategy
1. **Unified Pipeline**: All models use identical feature engineering for fair comparison
2. **Time-Based Split**: Last 20% of data used for testing (maintains temporal order)
3. **Automatic Scaling**: Features are standardized for models that need it
4. **Quick Training**: Models are configured for speed, not maximum accuracy

## How to Use It

### 1. Prepare Your Data
- **Format**: CSV or Excel file
- **Structure**: 
  - First column: Dates (YYYY-MM-DD format)
  - Last column: Target values (what you want to predict)
  - Middle columns: Optional features (e.g., marketing spend, weather)
- **Requirements**:
  - Consecutive daily dates
  - No missing dates in the middle
  - File size ≤ 1 MB
  - ≤ 10,000 rows

### 2. Add Future Dates for Forecasting
To get predictions, append future dates to your data:
```csv
2024-01-01,100,feature1,feature2
2024-01-02,105,feature1,feature2
...
2024-01-31,,feature1,feature2  # Leave target blank for future dates
2024-02-01,,feature1,feature2
```

### 3. Run the Application
```bash
# Install dependencies
python -m pip install -r requirements.txt

# Launch the app
streamlit run streamlit_app.py
```

### 4. Upload and Configure
1. **Upload your CSV/Excel file**
2. **Review data validation** - check for any warnings
3. **Configure options**:
   - Enable/disable trend fitting
   - Adjust feature engineering parameters
   - Select specific models to test
4. **Run the analysis**

### 5. Interpret Results
- **Performance Table**: Compare RMSE scores across models
- **Forecast Plot**: Visualize historical data and predictions
- **Feature Importance**: See which features drive predictions
- **Download Results**: Export forecasts and model performance

## How NOT to Use It

### Don't Use For:
- **High-frequency data** (hourly, minute-level) - designed for daily data only
- **Very long time series** (>10,000 rows) - performance degrades
- **Critical business decisions** without validation - this is a quick benchmark tool
- **Complex multivariate forecasting** - focus is on univariate with optional features
- **Production systems** - designed for exploration and quick insights

### Common Mistakes:
- **Inconsistent date formats** - use YYYY-MM-DD consistently
- **Missing dates** - fill gaps or use complete consecutive dates
- **Too many features** - stick to 5-10 meaningful features max
- **Ignoring warnings** - always check data validation messages
- **Overfitting** - don't use all available data for training

### Data Quality Issues:
- **Outliers** - extreme values can skew results
- **Seasonal breaks** - holidays, events that break patterns
- **Structural changes** - business model changes, new products
- **Missing values** - handle before uploading

## Technical Details

### Dependencies
- **Core ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, streamlit
- **Time Series**: statsmodels (for seasonal detection)
- **Data I/O**: openpyxl, xlrd

### Performance Characteristics
- **Training Time**: 5-30 seconds for typical datasets
- **Memory Usage**: Low (< 100MB for most datasets)
- **Scalability**: Best for 100-5,000 rows, degrades beyond 10,000

### Model Limitations
- **No Deep Learning**: Focuses on interpretable, fast models
- **No ARIMA**: Uses ML approach instead of statistical time series
- **No Cross-Validation**: Uses simple train/test split for speed

## Sample Datasets

The repository includes several sample datasets to test the tool:
- `dataset_1_no_features.csv` - Basic time series
- `dataset_2_one_feature.csv` - Time series with one external feature
- `dataset_3_two_features.csv` - Time series with two external features
- `dataset_4_three_features.csv` - Time series with three external features
- `dataset_5_retail_scenario.csv` - Retail sales example
- `train.csv` - Training dataset example

## Best Practices

1. **Start Simple**: Begin with basic features, add complexity gradually
2. **Validate Assumptions**: Check if your data has the patterns you expect
3. **Compare Models**: Don't rely on a single model - compare multiple approaches
4. **Check Residuals**: Look for patterns in prediction errors
5. **Update Regularly**: Retrain models as new data becomes available
6. **Domain Knowledge**: Use business understanding to interpret results

## Limitations and Caveats

- **No Guarantees**: This is a tool for exploration, not guaranteed accuracy
- **Assumes Stationarity**: Works best when patterns are relatively stable
- **Limited to Daily**: Not designed for other time frequencies
- **Quick and Dirty**: Prioritizes speed over perfect optimization
- **No External Validation**: Results are based on internal test split only

## Getting Help

If you encounter issues:
1. Check the data validation warnings
2. Ensure your data format matches requirements
3. Try with sample datasets first
4. Check that all dependencies are installed correctly

This tool is designed to give you quick insights into your time series data. Use it for exploration, benchmarking, and initial forecasting - but always validate results with domain knowledge and additional analysis.
