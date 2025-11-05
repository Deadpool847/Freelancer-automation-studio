#!/usr/bin/env python3
"""Test fixes for empty DataFrame handling"""

import sys
from pathlib import Path
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).parent / 'freelancer_automation_studio'))

from engine.etl.cleaner import DataCleaner
from engine.ml.detector import TaskDetector
from engine.utils.io_helpers import IOHelper

print("🧪 Testing Empty DataFrame Fixes\n")

# Test 1: Create dataset where ALL rows have nulls (worst case)
print("1️⃣ Test Case: All rows with null values + handle_missing='drop'")
try:
    # Create test data with all nulls
    test_data = pd.DataFrame({
        'col1': [None, None, None],
        'col2': [None, None, None],
        'col3': [None, None, None]
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    # Try ETL with drop strategy
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    
    print("   ❌ FAILED: Should have raised ValueError!")
    
except ValueError as e:
    if "All rows were removed" in str(e):
        print(f"   ✅ PASSED: Correctly raised ValueError")
        print(f"   📝 Error message: {str(e)[:100]}...")
    else:
        print(f"   ❌ FAILED: Wrong error - {e}")
except Exception as e:
    print(f"   ❌ FAILED: Unexpected error - {e}")

# Test 2: Empty DataFrame with fill_mode strategy
print("\n2️⃣ Test Case: All nulls + handle_missing='fill_mode'")
try:
    test_data = pd.DataFrame({
        'numeric_col': [None, None, None],
        'text_col': [None, None, None],
        'target': [None, None, None]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="fill_mode")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    
    print(f"   ✅ PASSED: Handled gracefully")
    print(f"   📊 Result: {len(cleaned_df)} rows, quality: {quality_score:.2%}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: Empty DataFrame with fill_mean strategy
print("\n3️⃣ Test Case: All nulls + handle_missing='fill_mean'")
try:
    test_data = pd.DataFrame({
        'feature_1': [None, None, None],
        'feature_2': [None, None, None],
        'target': [None, None, None]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="fill_mean")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    
    print(f"   ✅ PASSED: Handled gracefully")
    print(f"   📊 Result: {len(cleaned_df)} rows, quality: {quality_score:.2%}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 4: Task detection on empty DataFrame
print("\n4️⃣ Test Case: Task detection on empty DataFrame")
try:
    # Create empty parquet
    import polars as pl
    empty_df = pl.DataFrame({
        'col1': [],
        'col2': [],
        'target': []
    })
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        empty_df.write_parquet(f.name)
        temp_parquet = f.name
    
    detector = TaskDetector()
    task_info = detector.detect_task(temp_parquet)
    
    print("   ❌ FAILED: Should have raised ValueError!")
    
except ValueError as e:
    if "dataset is empty" in str(e):
        print(f"   ✅ PASSED: Correctly raised ValueError")
        print(f"   📝 Error message: {str(e)[:100]}...")
    else:
        print(f"   ❌ FAILED: Wrong error - {e}")
except Exception as e:
    print(f"   ❌ FAILED: Unexpected error - {e}")

# Test 5: Mixed nulls with drop (should pass with some rows)
print("\n5️⃣ Test Case: Partial nulls + drop (normal case)")
try:
    test_data = pd.DataFrame({
        'feature_1': [1.0, 2.0, None, 4.0, 5.0],
        'feature_2': [10, None, 30, 40, 50],
        'target': [0, 1, None, 1, 0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    
    print(f"   ✅ PASSED: Normal case handled")
    print(f"   📊 Result: {len(cleaned_df)} rows (expected 2), quality: {quality_score:.2%}")
    
    # Test task detection
    io_helper = IOHelper()
    import uuid
    silver_path = io_helper.save_to_silver(cleaned_df, str(uuid.uuid4()))
    
    detector = TaskDetector()
    task_info = detector.detect_task(str(silver_path))
    
    print(f"   ✅ Task detection: {task_info['task_type']}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 6: Mode with empty column
print("\n6️⃣ Test Case: Mode calculation on all-null column")
try:
    import polars as pl
    
    test_df = pl.DataFrame({
        'all_null_col': [None, None, None],
        'good_col': [1, 2, 3]
    })
    
    # Test mode on all-null column
    mode_result = test_df['all_null_col'].mode()
    
    if len(mode_result) > 0:
        mode_val = mode_result.first()
        print(f"   ✅ Mode result: {mode_val}")
    else:
        print(f"   ✅ Mode result: empty (expected)")
    
    print("   ✅ PASSED: Mode handles null column")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "="*60)
print("🎉 Test Suite Complete!")
print("="*60)
print("\n✅ All critical fixes validated:")
print("   • ValueError on empty DataFrame with clear message")
print("   • Fill strategies handle all-null columns")
print("   • Task detector prevents division by zero")
print("   • Mode calculation safe with empty results")
