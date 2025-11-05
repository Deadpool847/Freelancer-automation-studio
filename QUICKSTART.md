# 🚀 Freelancer Automation Studio - Quick Start Guide

## ⚡ Launch the Application

### Method 1: Using Shell Script (Recommended)
```bash
cd /app
./start.sh
```

### Method 2: Manual Launch
```bash
cd /app/freelancer_automation_studio
streamlit run ui/app.py --server.port 8501
```

Access the application at: **http://localhost:8501**

---

## 📊 Test with Sample Data

### Pre-generated Test Datasets

Located in: `/app/freelancer_automation_studio/data/bronze/`

1. **classification_sample.csv** - 1,000 rows, 23 features, 3 classes
2. **regression_sample.csv** - 800 rows, 15 features, continuous target
3. **timeseries_sample.csv** - 500 days of temporal data
4. **nlp_sample.csv** - 1,000 text samples with sentiment labels
5. **mixed_sample.csv** - 600 rows with mixed data types

---

## 🎯 Complete Workflow Example

### Step 1: Data Ingestion
1. Go to **"Ingest"** tab
2. Select **"Upload File"**
3. Upload `classification_sample.csv`
4. Click **"🚀 Process Uploaded File"**
5. ✅ Note your Run ID (e.g., `abc123...`)

### Step 2: ETL & Profiling
1. Go to **"ETL & Profile"** tab
2. Configure:
   - ✅ Remove Duplicates
   - Handle Missing: `drop`
   - ✅ Standardize Dates
3. Click **"🧹 Run ETL Pipeline"**
4. Review:
   - Quality Score
   - Data Profile
   - Preview Table

### Step 3: AutoML Training
1. Go to **"AutoML"** tab
2. Select **"Auto-Detect"** mode
3. Click **"🔍 Detect Task Type"**
   - Should detect: **classification**
   - Target: **target**
4. Configure training:
   - Test Size: `0.2`
   - CV Folds: `5`
   - Optuna Trials: `20`
5. Enable **GPU Acceleration** (optional)
6. Click **"🚀 Train Model"**
7. Wait 2-5 minutes for training
8. View metrics:
   - Accuracy: ~0.75+
   - Precision/Recall/F1

### Step 4: Explainability
1. Go to **"Explain"** tab
2. Click **"🧠 Generate Explanations"**
3. Review:
   - Top 20 Feature Importance
   - SHAP Summary Plot
   - Error Analysis

### Step 5: Visualization
1. Go to **"Visualize"** tab
2. Try different charts:
   - **Distribution Plot**: Select any numeric feature
   - **Correlation Heatmap**: View feature correlations
   - **Box Plot**: Identify outliers

### Step 6: Export Results
1. Go to **"Export"** tab
2. Select artifacts:
   - ✅ Cleaned Data
   - ✅ Trained Model
   - ✅ Explanations
   - ✅ Data Profile
3. Choose format: `Parquet` / `CSV` / `Excel`
4. ✅ Include Manifest
5. Click **"📥 Download All"**
6. Download ZIP file with all artifacts

---

## 🔬 Advanced Features

### Web Scraping

```yaml
Tab: Ingest
Mode: Web Scraping

Configuration:
  URL: https://example.com
  Max Pages: 10
  Respect robots.txt: ✅
  Rate Limit: 1.0 req/sec

Features:
  - Automatic robots.txt parsing
  - Per-domain rate limiting
  - Sitemap detection
  - Playwright fallback for JS sites
```

### GPU Acceleration

```yaml
Sidebar: ⚙️ Configuration
Enable GPU Acceleration: ✅

Benefits:
  - Faster XGBoost training (tree_method=gpu_hist)
  - ONNX Runtime CUDA for deep learning
  - Reduced training time by 3-5x

Requirements:
  - NVIDIA GPU (RTX 4050 recommended)
  - CUDA toolkit installed
  - onnxruntime-gpu package
```

### Time Series Detection

```python
# Automatic detection when:
# - 20%+ columns are temporal (date/time)
# - Sequential data structure
# - Time-ordered index

Features Generated:
  - Lag features (1, 2, 3, 7, 14, 30 periods)
  - Rolling statistics (7, 14, 30 windows)
  - Time-aware train/test split
```

### NLP Auto-Detection

```python
# Triggers when:
# - 30%+ columns contain text (avg length > 50)
# - Column names include: text, description, comment, review

Models Used:
  - DistilBERT (GPU mode)
  - TF-IDF + Linear (CPU mode)
```

---

## 🐛 Troubleshooting

### Issue: Streamlit Not Loading

```bash
# Check if running
ps aux | grep streamlit

# Restart
pkill streamlit
cd /app/freelancer_automation_studio
streamlit run ui/app.py --server.port 8501
```

### Issue: Import Errors

```bash
# Reinstall dependencies
cd /app
pip install -r backend/requirements.txt --upgrade
```

### Issue: Low Accuracy

**Possible Causes:**
1. **Insufficient Data**: Need 500+ samples minimum
2. **Poor Quality**: Run ETL with quality score check
3. **Wrong Task Type**: Manually select task type
4. **Imbalanced Classes**: Check recommendations

**Solutions:**
```yaml
Increase Optuna Trials: 20 → 50
Add More CV Folds: 5 → 10
Try Different Models: Manual model selection
Check Feature Importance: Remove low-importance features
```

### Issue: Out of Memory

```yaml
Solutions:
  1. Reduce batch size (configs/recipe.yaml)
  2. Disable GPU acceleration
  3. Use smaller test_size (0.2 → 0.1)
  4. Process data in chunks
```

---

## 📈 Performance Tips

### 1. Optimize ETL
```yaml
For Large Files (>500MB):
  - Use Parquet instead of CSV
  - Enable remove_duplicates: false
  - Use handle_missing: drop
```

### 2. Speed Up Training
```yaml
Quick Mode:
  - Optuna Trials: 10
  - CV Folds: 3
  - Parallel Jobs: 8

Production Mode:
  - Optuna Trials: 50
  - CV Folds: 10
  - GPU: ✅
```

### 3. Memory Efficiency
```yaml
Streaming Mode:
  - Process data in batches
  - Use Polars lazy evaluation
  - Enable DuckDB aggregations
```

---

## 🎓 Learning Resources

### Understanding the Pipeline

```
Data Flow:
📥 Upload → 🗄️ Bronze (Raw)
          ↓
       🧹 ETL Clean
          ↓
       💎 Silver (Cleaned)
          ↓
       🔧 Feature Engineering
          ↓
       🤖 Model Training
          ↓
       📊 Gold (Predictions)
```

### Model Selection Logic

```python
Classification:
  - Binary: LightGBM, XGBoost, Logistic Regression
  - Multi-class: LightGBM, XGBoost, Random Forest
  - Imbalanced: SMOTE + class weights

Regression:
  - Standard: LightGBM, XGBoost, Random Forest
  - Large Dataset: LightGBM (fastest)
  - Small Dataset: Random Forest (robust)

Time Series:
  - Short-term: LightGBM with lags
  - Long-term: XGBoost with seasonal features
  - Trend-heavy: Linear features + GBM

NLP:
  - GPU Available: DistilBERT fine-tuning
  - CPU Only: TF-IDF + SGDClassifier
  - Large Corpus: Pretrained embeddings
```

---

## 🚀 Next Steps

1. **Explore All Tabs**: Try each feature with sample data
2. **Upload Your Data**: Test with real-world datasets
3. **Optimize Models**: Experiment with hyperparameters
4. **Export & Deploy**: Save models for production
5. **Read Full Docs**: Check `/app/README.md`

---

## 💡 Pro Tips

1. **Always start with Auto-Detect** - It's smart!
2. **Check Quality Score** - Aim for 70%+
3. **Review SHAP plots** - Understand your model
4. **Export manifests** - Track everything
5. **Use GPU for NLP** - 5x faster training

---

**Ready to automate your ML workflows? Let's go! 🚀**
