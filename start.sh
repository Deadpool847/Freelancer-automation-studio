#!/bin/bash
# Freelancer Automation Studio Launcher

echo "🚀 Starting Freelancer Automation Studio..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r backend/requirements.txt

# Generate test data
echo "🧪 Generating test datasets..."
python freelancer_automation_studio/test_data_generator.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting Streamlit UI..."
echo "📍 Access at: http://localhost:8501"
echo ""

# Start Streamlit
cd freelancer_automation_studio
streamlit run ui/app.py --server.port 8501 --server.headless true