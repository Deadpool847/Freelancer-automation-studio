# 🏗️ Freelancer Automation Studio - Architecture Documentation

## 🎯 System Overview

Freelancer Automation Studio is a production-grade, fully local ML automation platform that handles the complete machine learning lifecycle from data ingestion to model deployment.

---

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (Port 8501)                     │
│                         Main Interface                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (Optional)                    │
│              RESTful API for Programmatic Access                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   INGESTION  │    │     ETL      │    │   FEATURES   │
│   ENGINE     │───▶│   ENGINE     │───▶│   ENGINE     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
    Bronze              Silver              Feature
     Layer              Layer               Store
                                                │
                                                ▼
                        ┌──────────────────────────────┐
                        │     ML ENGINE (Auto)         │
                        │  - Task Detection            │
                        │  - Model Training            │
                        │  - Hyperparameter Tuning     │
                        └────────────┬─────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │ EXPLAINER   │  │ VISUALIZER  │  │  REGISTRY   │
            │   (SHAP)    │  │  (Plotly)   │  │  (SQLite)   │
            └─────────────┘  └─────────────┘  └─────────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                            ┌─────────────────┐
                            │   EXPORT &      │
                            │   ARTIFACTS     │
                            └─────────────────┘
```

---

## 🔧 Component Details

### 1. **UI Layer (Streamlit)**

**Purpose**: User-facing interface for all operations

**Features**:
- 6 Interactive tabs (Ingest, ETL, AutoML, Explain, Visualize, Export)
- Real-time progress tracking
- Configuration sidebar
- Download buttons for artifacts

**Tech Stack**:
- Streamlit 1.51.0
- Plotly for visualizations
- Custom CSS for styling

---

### 2. **Ingestion Engine**

**Purpose**: Data acquisition from multiple sources

#### **Upload Module**
```python
Supported Formats:
  - CSV, Excel (xlsx/xls)
  - Parquet (Apache Arrow)
  - JSON (structured/nested)

Process:
  1. File validation
  2. Format detection
  3. Conversion to Parquet
  4. Storage in Bronze layer
```

#### **Web Scraping Module**
```python
Features:
  - robots.txt compliance (RobotFileParser)
  - Per-domain rate limiting
  - Sitemap parsing
  - Playwright fallback for JS

Tech:
  - httpx (HTTP client)
  - selectolax (Fast HTML parser)
  - Playwright (JS rendering)
  
Rate Limiting:
  - Configurable req/sec per domain
  - Respects Crawl-delay directive
  - Exponential backoff on errors
```

---

### 3. **ETL Engine**

**Purpose**: Data cleaning, transformation, and quality assurance

#### **Core Components**

**Data Cleaner** (`engine/etl/cleaner.py`):
```python
Operations:
  1. Duplicate Detection
     - SimHash for fuzzy matching
     - Exact row deduplication
  
  2. Missing Value Handling
     - drop: Remove rows with nulls
     - fill_mean: Numeric imputation
     - fill_median: Robust imputation
     - fill_mode: Categorical imputation
  
  3. Type Inference
     - Numeric: int/float detection
     - Categorical: Low cardinality strings
     - Temporal: Date/time parsing
     - Text: High-entropy strings
  
  4. Standardization
     - Column name normalization
     - Date format unification
     - Text cleaning (strip, lowercase)
```

**Data Profiler** (`engine/etl/cleaner.py`):
```python
Metrics:
  - Row/Column counts
  - Null percentages
  - Numeric stats (mean, std, min, max, median)
  - Categorical stats (unique count, top values)
  - Memory usage
  - Quality score (0-1)

Tech:
  - Polars for DataFrame operations
  - DuckDB for SQL-based profiling
  - PyArrow for zero-copy transfers
```

**Quality Scoring**:
```python
Formula:
  quality_score = (completeness * 0.4) + (missing_ratio * 0.6)

Where:
  - completeness = rows_after / rows_before
  - missing_ratio = 1 - (null_cells / total_cells)
```

---

### 4. **Feature Engineering Engine**

**Purpose**: Transform cleaned data into ML-ready features

#### **Feature Builder** (`engine/features/builder.py`)

**Operations**:
```python
1. Feature Extraction:
   - Datetime → [year, month, day, dayofweek]
   - Text → TF-IDF vectors (for NLP)
   - Categorical → Label encoding

2. Feature Scaling:
   - StandardScaler for numeric features
   - Preserves interpretability for tree models

3. Train/Test Splitting:
   - Stratified split for classification
   - Time-aware split for time series
   - Random split for regression

4. Time Series Features (if applicable):
   - Lag features: [1, 2, 3, 7, 14, 30]
   - Rolling stats: mean, std (7, 14, 30 windows)
   - Trend/seasonality extraction
```

**Tech**:
- scikit-learn (StandardScaler, LabelEncoder)
- Custom lag feature generation
- Memory-efficient pandas operations

---

### 5. **ML Engine (Core Intelligence)**

#### **Task Detector** (`engine/ml/detector.py`)

**Auto-Detection Logic**:
```python
Decision Tree:

1. Check for Text Columns (30%+ text):
   YES → Task = NLP
   NO  → Continue

2. Check for Image Columns:
   YES → Task = Computer Vision
   NO  → Continue

3. Check for Temporal Columns (20%+ dates):
   YES → Task = Time Series
   NO  → Continue

4. Check Target Column Type:
   Categorical (cardinality < 10%) → Classification
   Numeric (cardinality > 10%)     → Regression
   None                            → Clustering

5. Identify Target Column:
   - Search for: target, label, class, outcome, y
   - Default: Last column

6. Generate Recommendations:
   - Model suggestions
   - Data quality warnings
   - Feature engineering tips
```

#### **Model Trainer** (`engine/ml/trainer.py`)

**Classification Pipeline**:
```python
Models:
  1. LightGBM:
     - Fastest training
     - Handles missing values
     - Native categorical support
     - Best for: Large datasets (>10K rows)
  
  2. XGBoost:
     - GPU acceleration (tree_method=gpu_hist)
     - Regularization (L1/L2)
     - Best for: Medium datasets, GPU available
  
  3. Random Forest:
     - Robust to overfitting
     - No hyperparameter tuning needed
     - Best for: Small datasets (<5K rows)

Hyperparameter Optimization:
  - Optuna TPE Sampler
  - Cross-validation (stratified k-fold)
  - Search space:
    * n_estimators: [50, 500]
    * max_depth: [3, 15]
    * learning_rate: [0.01, 0.3] (log scale)
    * subsample: [0.6, 1.0]
    * colsample_bytree: [0.6, 1.0]

Metrics:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-Score (weighted)
  - ROC-AUC (binary only)
```

**Regression Pipeline**:
```python
Models:
  1. LightGBM Regressor
  2. XGBoost Regressor
  3. Random Forest Regressor

Optimization:
  - Minimize: Mean Squared Error
  - Same hyperparameter search as classification

Metrics:
  - MSE (Mean Squared Error)
  - RMSE (Root MSE)
  - MAE (Mean Absolute Error)
  - R² Score
```

**Time Series Pipeline**:
```python
Approach:
  - Convert to supervised learning
  - Add lag features
  - Use time-aware train/test split
  - GBM models with temporal features

Split Strategy:
  - Last 20% for testing
  - No shuffling
  - Respects temporal order
```

**NLP Pipeline**:
```python
GPU Mode:
  - Model: DistilBERT
  - Fine-tuning with Hugging Face Transformers
  - Mixed precision (fp16) for memory efficiency

CPU Mode:
  - TF-IDF vectorization (max_features=5000)
  - SGDClassifier (fast, linear)
  - Good for: Document classification

Preprocessing:
  - Tokenization
  - Lowercase
  - Remove special chars
  - Max length: 512 tokens
```

#### **Model Registry** (`engine/ml/registry.py`)

**Storage**:
```python
Backend: SQLite + Filesystem

Schema:
  models table:
    - model_id (PK): {run_id}_{task_type}
    - run_id: UUID
    - task_type: classification/regression/etc
    - model_path: /data/models/{model_id}.joblib
    - metrics: JSON (accuracy, precision, etc)
    - created_at: ISO timestamp

Operations:
  - save_model(model, run_id, task_type, metrics)
  - load_model(run_id, task_type)
  - get_model_info(run_id, task_type)
  - list_models(limit=50)
```

**Versioning**:
- Each run creates new model version
- Immutable storage
- Full reproducibility via manifests

---

### 6. **Explainability Engine**

#### **Model Explainer** (`engine/explain/explainer.py`)

**SHAP Integration**:
```python
Explainers:
  - TreeExplainer: For LightGBM, XGBoost, RF
  - KernelExplainer: For any model (slower)

Outputs:
  1. Feature Importance:
     - Top 20 features by impact
     - Sorted by mean |SHAP value|
  
  2. SHAP Summary Plot:
     - Beeswarm plot showing feature distributions
     - Color-coded by feature value
     - PNG export for reports
  
  3. Error Analysis:
     - Per-class error rates (classification)
     - Residual statistics (regression)
     - Identify systematic biases

Performance:
  - Max 100 samples for SHAP (configurable)
  - Caching for repeated explanations
  - Parallel computation
```

---

### 7. **Visualization Engine**

#### **Chart Generator** (`engine/viz/charts.py`)

**Interactive Charts**:
```python
1. Distribution Plot:
   - Histogram (numeric)
   - Bar chart (categorical)
   - Auto-bin calculation

2. Correlation Heatmap:
   - Pearson correlation
   - Numeric features only
   - Diverging colorscale (RdBu)

3. Time Series Plot:
   - Line chart with markers
   - Date-aware x-axis
   - Hover annotations

4. Scatter Matrix:
   - Pairwise scatter plots
   - Lower triangle only
   - Max 5 features

5. Box Plot:
   - Quartiles + outliers
   - Whiskers at 1.5*IQR
   - Identify anomalies

Tech:
  - Plotly (interactive)
  - Responsive layout
  - Export to PNG/HTML
```

---

### 8. **Manifest & Tracking System**

#### **Manifest Manager** (`engine/utils/manifest.py`)

**Purpose**: Full reproducibility and audit trail

**Tracked Metadata**:
```python
{
  "run_id": "uuid-v4",
  "mode": "upload | scrape",
  "source": "filename or URL",
  "config": {
    "etl_rules": {...},
    "model_params": {...}
  },
  "artifacts": {
    "bronze": "/path/to/raw.parquet",
    "silver": "/path/to/cleaned.parquet",
    "model": "/path/to/model.joblib",
    "report": "/path/to/report.json"
  },
  "timings": {
    "ingest": 1.2,
    "etl": 3.5,
    "training": 120.3,
    "total": 125.0
  },
  "status": "created | running | completed | failed",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:02:05"
}
```

**Hash Verification**:
- SHA-256 hash of input data
- Ensures data integrity
- Detect accidental modifications

---

### 9. **I/O & Storage System**

#### **Data Layers (Medallion Architecture)**

```python
Bronze Layer:
  - Raw, unprocessed data
  - Preserves original format
  - Immutable after ingestion
  - Compression: Snappy

Silver Layer:
  - Cleaned, standardized data
  - Schema-validated
  - Quality-scored
  - Optimized for queries

Gold Layer:
  - Business-ready data
  - Aggregations applied
  - Predictions included
  - Consumption-optimized

Feature Store:
  - Engineered features
  - Cached transformations
  - Reusable across runs

Models:
  - Serialized models (joblib)
  - Metadata in SQLite
  - Version-controlled

Reports:
  - JSON exports
  - Visualizations (PNG/HTML)
  - Manifest files
```

---

## 🔒 Security & Compliance

### Data Privacy
- ✅ **100% Local Execution**: No external API calls (except optional scraping)
- ✅ **No Telemetry**: Zero usage tracking
- ✅ **File Encryption**: Support for encrypted parquet files

### Web Scraping Compliance
- ✅ **robots.txt Respect**: Mandatory parsing
- ✅ **Rate Limiting**: Per-domain throttling
- ✅ **Crawl-delay**: Honor server directives
- ✅ **User-Agent**: Identifiable & respectful

### Access Control
- ✅ **Local-only**: No network exposure by default
- ✅ **File Permissions**: Proper Unix permissions
- ✅ **SQLite Security**: Database encryption support

---

## ⚡ Performance Optimizations

### Memory Efficiency
```python
1. Polars DataFrame:
   - Apache Arrow backend
   - Zero-copy operations
   - Memory-mapped files
   - 5-10x faster than pandas

2. Lazy Evaluation:
   - Query optimization
   - Predicate pushdown
   - Projection pruning

3. Streaming:
   - Batch processing
   - Chunked reading
   - Generator patterns
```

### Computation Speed
```python
1. Parallel Processing:
   - Polars native parallelism
   - scikit-learn n_jobs
   - Optuna parallel trials

2. GPU Acceleration:
   - XGBoost tree_method=gpu_hist
   - ONNX Runtime CUDA
   - PyTorch for deep learning

3. Compiled Code:
   - Polars (Rust-based)
   - LightGBM (C++)
   - XGBoost (C++)
```

### Disk I/O
```python
1. Parquet Format:
   - Columnar storage
   - Efficient compression
   - Fast reads/writes

2. DuckDB Integration:
   - In-process OLAP
   - SQL on Parquet
   - No serialization overhead
```

---

## 🧪 Testing & Validation

### Unit Tests
- Component-level testing
- Mock data generation
- Edge case coverage

### Integration Tests
- End-to-end pipelines
- Multi-format support
- Error handling

### Performance Tests
- Benchmark datasets
- Memory profiling
- Speed regression checks

---

## 📦 Deployment Options

### Local Development
```bash
streamlit run ui/app.py --server.port 8501
```

### Production Deployment
```bash
# Supervisor for process management
supervisord -c supervisord.conf

# Nginx reverse proxy
upstream fas {
    server 127.0.0.1:8501;
}
```

### Docker (Optional)
```dockerfile
FROM python:3.11
COPY . /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "ui/app.py"]
```

---

## 🔮 Future Enhancements

### Planned Features
1. **Deep Learning Models**:
   - CNN for image classification
   - Transformer models for NLP
   - LSTM for time series

2. **AutoML v2**:
   - Neural architecture search
   - Meta-learning
   - Transfer learning

3. **Deployment**:
   - FastAPI model serving
   - REST API generation
   - Docker containerization

4. **Monitoring**:
   - Real-time dashboards
   - Model drift detection
   - Performance alerts

---

## 📚 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit 1.51 | Interactive web interface |
| API | FastAPI 0.110 | RESTful endpoints |
| Data | Polars 1.35 | Fast DataFrame operations |
| SQL | DuckDB 0.10 | In-process analytics |
| ML | LightGBM 4.6 | Gradient boosting (tabular) |
| ML | XGBoost 3.1 | Gradient boosting (GPU) |
| ML | scikit-learn 1.4 | Classical ML algorithms |
| Explain | SHAP 0.49 | Model interpretability |
| Viz | Plotly 5.18 | Interactive charts |
| Scrape | httpx 0.26 | HTTP client |
| Scrape | selectolax 0.3 | Fast HTML parsing |
| Storage | SQLite 3.x | Metadata & registry |
| Format | Parquet | Columnar data storage |
| Scheduler | APScheduler 3.10 | Job scheduling |
| Logging | Loguru 0.7 | Structured logging |

---

**Built for Performance. Designed for Simplicity. Optimized for Accuracy.**
