import pandas as pd
import data_io
import data_utils

print("Testing date conversion fix...")

# Create test data with DD/MM/YYYY format
test_df = pd.DataFrame({
    'Date': ['01/01/2023', '02/01/2023', '03/01/2023', '04/01/2023', '05/01/2023'],
    'Temperature': [10, 12, 15, 18, 20]
})

print("Original data:")
print(test_df)
print(f"Date column type: {type(test_df['Date'].iloc[0])}")

# Test data_io processing
print("\n--- Testing data_io processing ---")
result_df, file_info = data_io.load_data_with_checklist(test_df)

print("After data_io processing:")
print(result_df)
print(f"Date column type: {type(result_df['Date'].iloc[0])}")
print(f"Detected format: {file_info.get('detected_date_format')}")

# Test data_utils processing
print("\n--- Testing data_utils processing ---")
series, load_meta = data_utils.prepare_series_from_dataframe(result_df, file_info)

print("After data_utils processing:")
print(series.head())
print(f"ds column type: {type(series['ds'].iloc[0])}")
print(f"Trailing missing count: {load_meta.get('trailing_missing_count')}")

print("\n✅ Test completed successfully!")
