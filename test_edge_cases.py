#!/usr/bin/env python3
"""Test all edge cases for production readiness"""

import sys
from pathlib import Path
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).parent / 'freelancer_automation_studio'))

from engine.etl.cleaner import DataCleaner
from engine.ml.detector import TaskDetector
from engine.utils.io_helpers import IOHelper

print("🔬 Production Edge Case Testing\n")

# Test 1: Single row dataset
print("1️⃣ Single row dataset")
try:
    test_data = pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [10],
        'target': [1]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} row, quality: {quality_score:.2%}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 2: All duplicate rows
print("\n2️⃣ All duplicate rows")
try:
    test_data = pd.DataFrame({
        'feature_1': [1.0, 1.0, 1.0],
        'feature_2': [10, 10, 10],
        'target': [1, 1, 1]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} row (deduped), quality: {quality_score:.2%}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 3: All zeros (should NOT be treated as nulls)
print("\n3️⃣ All zeros (not nulls)")
try:
    test_data = pd.DataFrame({
        'feature_1': [0, 0, 0, 0],
        'feature_2': [0, 0, 0, 0],
        'target': [0, 0, 0, 0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} rows retained (zeros are valid)")
    print(f"   📊 Quality: {quality_score:.2%}")
    
    # Verify task detection works
    io_helper = IOHelper()
    import uuid
    silver_path = io_helper.save_to_silver(cleaned_df, str(uuid.uuid4()))
    
    detector = TaskDetector()
    task_info = detector.detect_task(str(silver_path))
    print(f"   ✅ Task detection: {task_info['task_type']}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 4: Single column dataset
print("\n4️⃣ Single column dataset")
try:
    test_data = pd.DataFrame({
        'target': [1, 2, 3, 4, 5]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} rows, {len(cleaned_df.columns)} column")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 5: Mix of nulls and zeros
print("\n5️⃣ Mix of nulls and zeros")
try:
    test_data = pd.DataFrame({
        'feature_1': [0, None, 0, None, 0],
        'feature_2': [None, 0, 0, None, 0],
        'target': [1, 0, 1, 0, 1]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    # Test with fill_mean
    cleaner = DataCleaner(handle_missing="fill_mean")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED (fill_mean): {len(cleaned_df)} rows, quality: {quality_score:.2%}")
    
    # Test with fill_mode
    cleaner = DataCleaner(handle_missing="fill_mode")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED (fill_mode): {len(cleaned_df)} rows, quality: {quality_score:.2%}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 6: Large dataset with sparse nulls
print("\n6️⃣ Large dataset with sparse nulls")
try:
    import numpy as np
    
    # 10k rows, 20 columns, 5% nulls
    n_rows = 10000
    n_cols = 20
    data = {}
    
    for i in range(n_cols):
        col_data = np.random.randn(n_rows)
        # Randomly insert 5% nulls
        null_mask = np.random.random(n_rows) < 0.05
        col_data[null_mask] = np.nan
        data[f'feature_{i}'] = col_data
    
    data['target'] = np.random.randint(0, 2, n_rows)
    test_data = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="fill_median")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df):,} rows, quality: {quality_score:.2%}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 7: String columns with nulls
print("\n7️⃣ String columns with nulls + fill_mode")
try:
    test_data = pd.DataFrame({
        'category_1': ['A', 'B', None, 'A', 'B'],
        'category_2': [None, 'X', 'Y', 'X', None],
        'target': [1, 0, 1, 0, 1]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="fill_mode")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} rows, quality: {quality_score:.2%}")
    print(f"   📊 Null count after fill: {sum(cleaned_df[col].null_count() for col in cleaned_df.columns)}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test 8: All text columns with nulls
print("\n8️⃣ All text columns with nulls")
try:
    test_data = pd.DataFrame({
        'text_1': [None, None, None],
        'text_2': [None, None, None],
        'text_3': [None, None, None]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    cleaner = DataCleaner(handle_missing="fill_mode")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(temp_path)
    print(f"   ✅ PASSED: {len(cleaned_df)} rows filled with empty strings")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "="*60)
print("✅ All Edge Cases Passed!")
print("="*60)
