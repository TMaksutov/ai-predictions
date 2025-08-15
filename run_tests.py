#!/usr/bin/env python3
"""
Comprehensive test runner for the time series forecasting project.
Provides different testing options and generates test reports.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        print(f"✅ {description} completed successfully in {end_time - start_time:.2f}s")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ {description} failed after {end_time - start_time:.2f}s")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def run_basic_tests():
    """Run basic test suite."""
    return run_command(
        ["python3", "-m", "pytest", "tests/", "-v"],
        "Basic test suite"
    )

def run_performance_tests():
    """Run performance tests."""
    return run_command(
        ["python3", "-m", "pytest", "tests/test_performance.py", "-v", "-m", "performance"],
        "Performance tests"
    )

def run_edge_case_tests():
    """Run edge case tests."""
    return run_command(
        ["python3", "-m", "pytest", "tests/test_edge_cases.py", "-v", "-m", "edge_case"],
        "Edge case tests"
    )

def run_sample_file_tests():
    """Run tests on sample files."""
    return run_command(
        ["python3", "-m", "pytest", "tests/test_sample_files.py", "-v"],
        "Sample file tests"
    )

def run_all_tests_with_coverage():
    """Run all tests with coverage report."""
    # First install coverage if not available
    try:
        import coverage
    except ImportError:
        print("Installing coverage...")
        subprocess.run(["pip3", "install", "--break-system-packages", "coverage"], check=True)
    
    return run_command(
        ["python3", "-m", "pytest", "tests/", "--cov=ts_core", "--cov-report=html", "--cov-report=term"],
        "All tests with coverage"
    )

def run_specific_test_file(test_file):
    """Run tests from a specific file."""
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return False
    
    return run_command(
        ["python3", "-m", "pytest", test_file, "-v"],
        f"Tests from {test_file}"
    )

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("GENERATING TEST REPORT")
    print("="*60)
    
    # Count test files
    test_files = list(Path("tests").glob("test_*.py"))
    print(f"Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"  - {tf.name}")
    
    # Count test data files
    data_files = list(Path("test_files").glob("*.csv"))
    print(f"\nFound {len(data_files)} test data files:")
    for df in data_files:
        print(f"  - {df.name}")
    
    # Check if all tests pass
    print("\nRunning quick test to check if all tests pass...")
    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "--tb=no", "-q"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("✅ All tests are passing!")
    else:
        print("❌ Some tests are failing. Run with -v for details.")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Test runner for time series forecasting project")
    parser.add_argument("--basic", action="store_true", help="Run basic test suite")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    parser.add_argument("--sample-files", action="store_true", help="Run sample file tests")
    parser.add_argument("--coverage", action="store_true", help="Run all tests with coverage")
    parser.add_argument("--file", type=str, help="Run tests from specific file")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Set PATH for pytest
    os.environ["PATH"] = "/home/ubuntu/.local/bin:" + os.environ.get("PATH", "")
    
    if args.all or not any([args.basic, args.performance, args.edge_cases, args.sample_files, args.coverage, args.file, args.report]):
        # Default: run all tests
        print("Running all tests...")
        success = True
        success &= run_basic_tests()
        success &= run_edge_case_tests()
        success &= run_sample_file_tests()
        success &= run_performance_tests()
        
        if success:
            print("\n🎉 All test suites completed successfully!")
        else:
            print("\n💥 Some test suites failed!")
            sys.exit(1)
    
    else:
        # Run specific test types
        success = True
        
        if args.basic:
            success &= run_basic_tests()
        
        if args.performance:
            success &= run_performance_tests()
        
        if args.edge_cases:
            success &= run_edge_case_tests()
        
        if args.sample_files:
            success &= run_sample_file_tests()
        
        if args.coverage:
            success &= run_all_tests_with_coverage()
        
        if args.file:
            success &= run_specific_test_file(args.file)
        
        if args.report:
            generate_test_report()
        
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()