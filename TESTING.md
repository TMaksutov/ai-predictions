# Testing Guide

This document provides comprehensive information about the testing setup for the time series forecasting project.

## Test Structure

The project contains a robust testing framework with the following components:

### Test Files

- **`tests/test_ts_core.py`** - Core functionality tests for the main forecasting functions
- **`tests/test_sample_files.py`** - Tests using various sample data files
- **`tests/test_edge_cases.py`** - Edge case and error condition tests
- **`tests/test_performance.py`** - Performance and stress testing

### Test Data Files

- **`test_files/daily.csv`** - Daily time series data
- **`test_files/hourly.csv`** - Hourly time series data
- **`test_files/minute_multi.csv`** - Multi-column minute-level data
- **`test_files/no_header.csv`** - CSV without headers
- **`test_files/weekly_multi.csv** - Weekly multi-column data
- **`test_files/monthly_data.csv`** - Monthly time series data
- **`test_files/quarterly_data.csv** - Quarterly business data
- **`test_files/irregular_data.csv** - Irregular interval data
- **`test_files/mixed_types.csv** - Data with mixed types and categories

## Running Tests

### Prerequisites

Install the required testing dependencies:

```bash
pip install pytest pandas numpy scikit-learn openpyxl xlrd
```

### Basic Test Execution

Run all tests:
```bash
python3 -m pytest tests/ -v
```

Run specific test file:
```bash
python3 -m pytest tests/test_ts_core.py -v
```

Run tests with coverage:
```bash
python3 -m pytest tests/ --cov=ts_core --cov-report=html
```

### Using the Test Runner

The project includes a comprehensive test runner script:

```bash
# Run all tests
python3 run_tests.py --all

# Run specific test types
python3 run_tests.py --basic
python3 run_tests.py --performance
python3 run_tests.py --edge-cases
python3 run_tests.py --sample-files

# Generate test report
python3 run_tests.py --report

# Run tests with coverage
python3 run_tests.py --coverage
```

## Test Categories

### 1. Core Functionality Tests (`test_ts_core.py`)

Tests the main forecasting and data loading functions:

- **Linear trend forecasting** - Tests basic forecasting functionality
- **Small sample fallback** - Tests handling of insufficient data
- **CSV loading** - Tests various CSV formats and edge cases
- **Interval detection** - Tests automatic frequency detection
- **Error handling** - Tests proper error responses

### 2. Sample File Tests (`test_sample_files.py`)

Tests the system with real-world data files:

- **Multiple frequencies** - Daily, hourly, weekly, monthly, quarterly
- **Different formats** - With/without headers, various separators
- **Data quality** - Handles missing values, invalid data gracefully

### 3. Edge Case Tests (`test_edge_cases.py`)

Tests error conditions and boundary cases:

- **Empty dataframes** - Proper error handling
- **Missing columns** - Column validation
- **All NaN values** - Data quality checks
- **Single data point** - Minimal data handling
- **Duplicate dates** - Data deduplication
- **Mixed data types** - Robust data parsing
- **Invalid horizons** - Parameter validation

### 4. Performance Tests (`test_performance.py`)

Tests system performance and scalability:

- **Large datasets** - Performance with 1000+ data points
- **Long horizons** - Forecasting 1000+ periods ahead
- **Memory efficiency** - Resource usage monitoring
- **Concurrent operations** - Multi-threading support
- **Numerical stability** - Extreme value handling
- **Consistency** - Deterministic results

## Test Configuration

### pytest.ini

The project includes a `pytest.ini` configuration file with:

- Test discovery patterns
- Markers for different test types
- Warning filters
- Output formatting options

### Test Markers

Use markers to run specific test categories:

```bash
# Run only performance tests
python3 -m pytest -m performance

# Run only edge case tests
python3 -m pytest -m edge_case

# Run all tests except slow ones
python3 -m pytest -m "not slow"
```

## Continuous Integration

The testing framework is designed to work with CI/CD pipelines:

- All tests must pass before merging
- Coverage reports are generated automatically
- Performance benchmarks are tracked
- Edge cases are thoroughly tested

## Adding New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Test Structure

```python
def test_functionality_description():
    """Brief description of what is being tested."""
    # Arrange - Set up test data
    df = pd.DataFrame(...)
    
    # Act - Execute the function
    result = function_under_test(df)
    
    # Assert - Verify the results
    assert len(result) == expected_length
    assert "expected_column" in result.columns
```

### Test Data

When adding new test data files:

1. Place in `test_files/` directory
2. Use descriptive names
3. Include various data types and edge cases
4. Document the expected behavior

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `PYTHONPATH` includes the project root
2. **Missing dependencies**: Install required packages
3. **Test failures**: Check test data and expected outputs
4. **Performance issues**: Monitor system resources

### Debug Mode

Run tests with verbose output for debugging:

```bash
python3 -m pytest tests/ -v -s --tb=long
```

## Test Results

Current test status: **43 tests passing**

- Core functionality: ✅ All passing
- Sample files: ✅ All passing  
- Edge cases: ✅ All passing
- Performance: ✅ All passing

## Contributing

When contributing to the project:

1. Write tests for new functionality
2. Ensure all existing tests pass
3. Add appropriate test markers
4. Update this documentation if needed
5. Run the full test suite before submitting

## Support

For testing-related issues:

1. Check the test output for specific error messages
2. Review the test data and expected results
3. Ensure all dependencies are properly installed
4. Consult the pytest documentation for advanced usage