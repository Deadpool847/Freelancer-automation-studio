#!/bin/bash
echo "🔍 Freelancer Automation Studio - System Check"
echo "=============================================="
echo ""

# Check Python version
echo "✓ Python Version:"
python3 --version

# Check key dependencies
echo ""
echo "✓ Key Dependencies:"
python3 -c "import polars; print(f'  - Polars: {polars.__version__}')"
python3 -c "import lightgbm; print(f'  - LightGBM: {lightgbm.__version__}')"
python3 -c "import xgboost; print(f'  - XGBoost: {xgboost.__version__}')"
python3 -c "import streamlit; print(f'  - Streamlit: {streamlit.__version__}')"
python3 -c "import shap; print(f'  - SHAP: {shap.__version__}')"

# Check directory structure
echo ""
echo "✓ Directory Structure:"
echo "  - Bronze: $(ls -1 /app/freelancer_automation_studio/data/bronze/ | wc -l) files"
echo "  - Silver: $(ls -1 /app/freelancer_automation_studio/data/silver/ 2>/dev/null | wc -l) files"
echo "  - Models: $(ls -1 /app/freelancer_automation_studio/data/models/ 2>/dev/null | wc -l) files"

# Check Streamlit process
echo ""
echo "✓ Streamlit Status:"
if ps aux | grep -v grep | grep streamlit > /dev/null; then
    echo "  - Running on port 8501 ✅"
else
    echo "  - Not running ⚠️"
fi

# Check test data
echo ""
echo "✓ Sample Datasets:"
ls -lh /app/freelancer_automation_studio/data/bronze/*.csv 2>/dev/null | awk '{print "  -", $9, "("$5")"}'

echo ""
echo "=============================================="
echo "✅ System Check Complete!"
echo ""
echo "🌐 Access UI at: http://localhost:8501"
echo "📖 Quick Start: /app/QUICKSTART.md"
echo "📚 Full Docs: /app/README.md"
