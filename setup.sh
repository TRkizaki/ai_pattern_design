#!/bin/bash

echo “🤖 Pattern Recognition Presentation Setup”
echo “==========================================”

# 仮想環境作成

echo “📦 Creating virtual environment…”
python -m venv venv

# 仮想環境のアクティベート（OS別）

if [[ “$OSTYPE” == “msys” || “$OSTYPE” == “win32” ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo “📥 Installing dependencies…”
pip install –upgrade pip
pip install -r requirements.txt

echo “✅ Setup complete!”
echo “”
echo “🚀 To start the presentation:”
echo “   For Streamlit: streamlit run streamlit_app.py”
echo “   For Jupyter: jupyter notebook notebooks/pattern_recognition_demo.ipynb”
echo “   For HTML: Open interactive_demo.html in your browser”

# =================================

# setup.bat (Windows用)

# =================================

@echo off
echo 🤖 Pattern Recognition Presentation Setup
echo ==========================================

echo 📦 Creating virtual environment…
