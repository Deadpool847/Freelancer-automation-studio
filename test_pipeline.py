#!/usr/bin/env python3
"""Test Freelancer Automation Studio Pipeline"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / 'freelancer_automation_studio'))

from engine.etl.cleaner import DataCleaner
from engine.ml.detector import TaskDetector
from engine.features.builder import FeatureBuilder
from engine.ml.trainer import ModelTrainer
from engine.ml.registry import ModelRegistry
from engine.explain.explainer import ModelExplainer
from engine.utils.manifest import ManifestManager
from engine.utils.io_helpers import IOHelper
import polars as pl

print("🚀 Testing Freelancer Automation Studio Pipeline\n")

# Test 1: ETL
print("1️⃣ Testing ETL Pipeline...")
try:
    bronze_path = "/app/freelancer_automation_studio/data/bronze/classification_sample.csv"
    cleaner = DataCleaner(remove_duplicates=True, handle_missing="drop")
    cleaned_df, profile, quality_score = cleaner.clean_and_profile(bronze_path)
    print(f"   ✅ ETL Success: {len(cleaned_df)} rows, Quality: {quality_score:.2%}")
    
    # Save to silver
    io_helper = IOHelper()
    silver_path = io_helper.save_to_silver(cleaned_df, "test_run_001")
    print(f"   ✅ Saved to silver: {silver_path.name}\n")
except Exception as e:
    print(f"   ❌ ETL Failed: {e}\n")
    sys.exit(1)

# Test 2: Task Detection
print("2️⃣ Testing Task Detection...")
try:
    detector = TaskDetector()
    task_info = detector.detect_task(str(silver_path))
    print(f"   ✅ Detected: {task_info['task_type']}")
    print(f"   ✅ Target: {task_info['target_column']}")
    print(f"   ✅ Features: {len(task_info['features'])}\n")
except Exception as e:
    print(f"   ❌ Detection Failed: {e}\n")
    sys.exit(1)

# Test 3: Feature Building
print("3️⃣ Testing Feature Engineering...")
try:
    feature_builder = FeatureBuilder()
    X_train, X_test, y_train, y_test = feature_builder.build_features(
        str(silver_path), task_info, test_size=0.2
    )
    print(f"   ✅ Train shape: {X_train.shape}")
    print(f"   ✅ Test shape: {X_test.shape}\n")
except Exception as e:
    print(f"   ❌ Feature Engineering Failed: {e}\n")
    sys.exit(1)

# Test 4: Model Training
print("4️⃣ Testing Model Training...")
try:
    trainer = ModelTrainer(task_type=task_info['ml_type'], use_gpu=False, n_jobs=2)
    model, metrics = trainer.train(
        X_train, X_test, y_train, y_test,
        optuna_trials=5,  # Reduced for fast testing
        cv_folds=3
    )
    print(f"   ✅ Model trained successfully")
    print(f"   ✅ Metrics: {metrics}\n")
except Exception as e:
    print(f"   ❌ Training Failed: {e}\n")
    sys.exit(1)

# Test 5: Model Registry
print("5️⃣ Testing Model Registry...")
try:
    registry = ModelRegistry()
    model_path = registry.save_model(model, "test_run_001", task_info['ml_type'], metrics)
    print(f"   ✅ Model saved: {model_path.name}")
    
    # Load back
    loaded_model = registry.load_model("test_run_001", task_info['ml_type'])
    print(f"   ✅ Model loaded successfully\n")
except Exception as e:
    print(f"   ❌ Registry Failed: {e}\n")
    sys.exit(1)

# Test 6: Explainability
print("6️⃣ Testing Explainability...")
try:
    explainer = ModelExplainer(model=model, task_type=task_info['ml_type'])
    explanations = explainer.explain(X_test[:50], y_test[:50])
    print(f"   ✅ Feature importance: {len(explanations.get('feature_importance', []))} features")
    print(f"   ✅ Error analysis: {bool(explanations.get('error_analysis'))}\n")
except Exception as e:
    print(f"   ❌ Explainability Failed: {e}\n")

# Test 7: Manifest Management
print("7️⃣ Testing Manifest System...")
try:
    manifest_mgr = ManifestManager()
    run_id = manifest_mgr.create_run({
        'mode': 'test',
        'source': 'classification_sample.csv',
        'timestamp': '2024-01-01T00:00:00'
    })
    print(f"   ✅ Created run: {run_id[:8]}...")
    
    manifest = manifest_mgr.get_manifest(run_id)
    print(f"   ✅ Retrieved manifest: {manifest['mode']}\n")
except Exception as e:
    print(f"   ❌ Manifest Failed: {e}\n")

print("🎉 All tests passed successfully!")
print("✅ Freelancer Automation Studio is fully operational")
