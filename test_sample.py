#!/usr/bin/env python3
import pandas as pd
import numpy as np
from modeling import UnifiedTimeSeriesTrainer

# Load the sample dataset
df = pd.read_csv('sample.csv')
print(f"Sample dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# Check the data structure
print(f"\nColumns: {list(df.columns)}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"Target 'y' - Min: {df['y'].min()}, Max: {df['y'].max()}, Mean: {df['y'].mean():.2f}")

# Check for missing values
missing_y = df['y'].isna().sum()
print(f"Missing 'y' values: {missing_y}")

# Check the last few rows
print(f"\nLast 10 rows:")
print(df[['ds', 'y']].tail(10))

# Try to run a simple benchmark
print(f"\nRunning benchmark on sample dataset...")
try:
    trainer = UnifiedTimeSeriesTrainer()
    results = trainer.benchmark_models(df, test_fraction=0.2)
    
    print(f"Benchmark completed. Got {len(results)} results.")
    
    if results:
        first_result = results[0]
        print(f"\nFirst result details:")
        print(f"Name: {first_result.get('name')}")
        print(f"RMSE: {first_result.get('rmse')}")
        print(f"MAPE: {first_result.get('mape')}")
        print(f"Accuracy %: {first_result.get('accuracy_pct')}")
        
        # Check test data
        test_df = first_result.get('test_df')
        if test_df is not None and not test_df.empty:
            print(f"Test data shape: {test_df.shape}")
            print(f"Test data 'y' values: {test_df['y'].tolist()}")
            print(f"Test data 'y' mean: {test_df['y'].mean()}")
            print(f"Test data 'y' std: {test_df['y'].std()}")
        else:
            print("No test data available")
            
    else:
        print("No results returned from benchmark")
        
except Exception as e:
    print(f"Benchmark failed with error: {e}")
    import traceback
    traceback.print_exc()
