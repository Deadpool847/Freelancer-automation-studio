#!/usr/bin/env python3
"""Generate sample test datasets for Freelancer Automation Studio"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.datasets import make_classification, make_regression

# Output directory
output_dir = Path(__file__).parent / "data" / "bronze"
output_dir.mkdir(parents=True, exist_ok=True)

print("🚀 Generating test datasets...\n")

# 1. Classification Dataset
print("1️⃣ Generating classification dataset...")
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

df_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(20)])
df_clf['target'] = y_clf
df_clf['category'] = np.random.choice(['A', 'B', 'C'], size=1000)
df_clf['date'] = pd.date_range(start='2023-01-01', periods=1000, freq='H')

clf_path = output_dir / 'classification_sample.csv'
df_clf.to_csv(clf_path, index=False)
print(f"   ✅ Saved: {clf_path}")
print(f"   📊 Shape: {df_clf.shape}")
print(f"   🎯 Classes: {df_clf['target'].nunique()}\n")

# 2. Regression Dataset
print("2️⃣ Generating regression dataset...")
X_reg, y_reg = make_regression(
    n_samples=800,
    n_features=15,
    n_informative=10,
    noise=10,
    random_state=42
)

df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(15)])
df_reg['target'] = y_reg
df_reg['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=800)

reg_path = output_dir / 'regression_sample.csv'
df_reg.to_csv(reg_path, index=False)
print(f"   ✅ Saved: {reg_path}")
print(f"   📊 Shape: {df_reg.shape}")
print(f"   📈 Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]\n")

# 3. Time Series Dataset
print("3️⃣ Generating time series dataset...")
dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
trend = np.linspace(100, 200, 500)
seasonality = 20 * np.sin(np.linspace(0, 4 * np.pi, 500))
noise = np.random.randn(500) * 5
values = trend + seasonality + noise

df_ts = pd.DataFrame({
    'date': dates,
    'value': values,
    'trend': trend,
    'seasonality': seasonality,
    'moving_avg_7': pd.Series(values).rolling(7).mean(),
    'moving_avg_30': pd.Series(values).rolling(30).mean()
})

ts_path = output_dir / 'timeseries_sample.csv'
df_ts.to_csv(ts_path, index=False)
print(f"   ✅ Saved: {ts_path}")
print(f"   📊 Shape: {df_ts.shape}")
print(f"   📅 Date range: {dates[0].date()} to {dates[-1].date()}\n")

# 4. NLP Dataset
print("4️⃣ Generating NLP dataset...")
sentences = [
    "This product is amazing and works perfectly!",
    "Terrible experience, would not recommend.",
    "Average quality, nothing special.",
    "Best purchase I've made this year!",
    "Complete waste of money and time.",
    "Good value for the price.",
    "Disappointed with the quality.",
    "Exceeded my expectations!"
]

sentiment_labels = [1, 0, 2, 1, 0, 1, 0, 1]  # 0=negative, 1=positive, 2=neutral

# Expand dataset
expanded_sentences = []
expanded_labels = []

for _ in range(125):
    for sent, label in zip(sentences, sentiment_labels):
        expanded_sentences.append(sent)
        expanded_labels.append(label)

df_nlp = pd.DataFrame({
    'text': expanded_sentences,
    'sentiment': expanded_labels,
    'length': [len(s) for s in expanded_sentences],
    'word_count': [len(s.split()) for s in expanded_sentences]
})

nlp_path = output_dir / 'nlp_sample.csv'
df_nlp.to_csv(nlp_path, index=False)
print(f"   ✅ Saved: {nlp_path}")
print(f"   📊 Shape: {df_nlp.shape}")
print(f"   💬 Sentiment distribution: {df_nlp['sentiment'].value_counts().to_dict()}\n")

# 5. Mixed Dataset (Complex)
print("5️⃣ Generating mixed/complex dataset...")
df_mixed = pd.DataFrame({
    'id': range(1, 601),
    'numeric_feature_1': np.random.randn(600),
    'numeric_feature_2': np.random.uniform(0, 100, 600),
    'categorical_1': np.random.choice(['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D'], 600),
    'categorical_2': np.random.choice(['Type_X', 'Type_Y', 'Type_Z'], 600),
    'text_field': np.random.choice([
        'Short text',
        'This is a medium length text with more words',
        'A very long text field containing lots of information that might be useful'
    ], 600),
    'date_created': pd.date_range(start='2020-01-01', periods=600, freq='D'),
    'missing_data': [np.nan if np.random.rand() < 0.2 else np.random.randn() for _ in range(600)],
    'target': np.random.choice([0, 1], 600)
})

mixed_path = output_dir / 'mixed_sample.csv'
df_mixed.to_csv(mixed_path, index=False)
print(f"   ✅ Saved: {mixed_path}")
print(f"   📊 Shape: {df_mixed.shape}")
print(f"   🔍 Features: {len(df_mixed.columns)} columns\n")

print("✨ All test datasets generated successfully!")
print(f"📁 Location: {output_dir}")
print("\n🎯 Ready to test in Streamlit UI!")