import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path('/workspace/test_files')
rng = np.random.default_rng(42)

def write_daily():
	# 10 daily points
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='D')
	t = np.arange(n)
	values = 50 + 0.8 * t + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, n)
	df = pd.DataFrame({'date': dates, 'value': np.round(values, 3)})
	(df.assign(date=df['date'].dt.strftime('%Y-%m-%d')))
	df.to_csv(BASE_DIR / 'daily.csv', index=False)


def write_hourly():
	# 10 hourly points
	n = 10
	dates = pd.date_range('2023-01-01 00:00:00', periods=n, freq='H')
	t = np.arange(n)
	values = 5 + np.log1p(t) + 2 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.1, n)
	df = pd.DataFrame({'date': dates, 'value': np.round(values, 3)})
	(df.assign(date=df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')))
	df.to_csv(BASE_DIR / 'hourly.csv', index=False)


def write_minute_multi():
	# 10 minute-level rows with features
	n = 10
	dates = pd.date_range('2023-01-01 00:00:00', periods=n, freq='T')
	t = np.arange(n)
	value = 20 + 0.2 * t + 3 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.1, n)
	feature1 = np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.01, n)
	feature2 = 100 * np.cos(2 * np.pi * t / 60) + rng.normal(0, 0.5, n)
	df = pd.DataFrame({
		'date': dates,
		'value': np.round(value, 3),
		'feature1': np.round(feature1, 4),
		'feature2': np.round(feature2, 3),
	})
	df.to_csv(BASE_DIR / 'minute_multi.csv', index=False)


def write_weekly_multi():
	# 10 weekly rows with features
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='W-SUN')
	t = np.arange(n)
	value = 100 + 5 * t + 20 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.5, n)
	feature1 = 0.1 * t + rng.normal(0, 0.01, n)
	feature2 = t + rng.normal(0, 0.1, n)
	df = pd.DataFrame({
		'date': dates,
		'value': np.round(value, 3),
		'feature1': np.round(feature1, 3),
		'feature2': np.round(feature2, 3),
	})
	df.to_csv(BASE_DIR / 'weekly_multi.csv', index=False)


def write_monthly():
	# 10 monthly rows
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='MS')
	t = np.arange(n)
	value = 200 + 10 * t + 30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.3, n)
	df = pd.DataFrame({'date': dates, 'value': np.round(value, 3)})
	df.to_csv(BASE_DIR / 'monthly_data.csv', index=False)


def write_quarterly():
	# 10 quarterly rows with revenue and profit
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='QS-JAN')
	t = np.arange(n)
	revenue = 1_000_000 + 50_000 * t + 200_000 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 5_000, n)
	profit = 0.15 * revenue + 10_000 * np.cos(2 * np.pi * t / 4) + rng.normal(0, 2_000, n)
	df = pd.DataFrame({
		'date': dates,
		'revenue': np.round(revenue).astype(int),
		'profit': np.round(profit).astype(int),
	})
	df.to_csv(BASE_DIR / 'quarterly_data.csv', index=False)


def write_irregular():
	# Use existing irregular dates, compute values
	irregular_dates = [
		'2023-01-01','2023-01-03','2023-01-07','2023-01-15','2023-01-30',
		'2023-02-15','2023-03-10','2023-03-21','2023-04-05','2023-05-01'
	]
	n = len(irregular_dates)
	t = np.arange(n)
	value = 10 + 2.5 * np.log1p(t) + 4 * (t ** 0.5) + rng.normal(0, 0.3, n)
	df = pd.DataFrame({'date': pd.to_datetime(irregular_dates), 'value': np.round(value, 3)})
	df.to_csv(BASE_DIR / 'irregular_data.csv', index=False)


def write_mixed_types():
	# 10 daily rows with categorical and text columns preserved
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='D')
	t = np.arange(n)
	value = 50 + 5 * np.log1p(t) + 8 * np.cos(2 * np.pi * t / 7) + rng.normal(0, 0.25, n)
	categories = ['A','B','C'] * (n // 3 + 1)
	categories = categories[:n]
	notes = [f"Entry {i+1}" for i in range(n)]
	df = pd.DataFrame({
		'date': dates,
		'value': np.round(value, 3),
		'category': categories,
		'notes': notes,
	})
	df.to_csv(BASE_DIR / 'mixed_types.csv', index=False)


def write_no_header():
	# 10 daily rows, no header
	n = 10
	dates = pd.date_range('2023-01-01', periods=n, freq='D')
	t = np.arange(n)
	value = 30 + 1.2 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, n)
	df = pd.DataFrame({'date': dates, 'value': np.round(value, 3)})
	df.to_csv(BASE_DIR / 'no_header.csv', index=False, header=False, date_format='%Y-%m-%d')


def main():
	write_daily()
	write_hourly()
	write_minute_multi()
	write_weekly_multi()
	write_monthly()
	write_quarterly()
	write_irregular()
	write_mixed_types()
	write_no_header()

if __name__ == '__main__':
	main()