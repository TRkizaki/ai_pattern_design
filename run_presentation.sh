#!/bin/bash

echo “🎓 Pattern Recognition Presentation Launcher”
echo “============================================”
echo “”
echo “Choose your presentation format:”
echo “1) 🌐 HTML Demo (Browser)”
echo “2) 📊 Streamlit Web App”
echo “3) 📚 Jupyter Notebook”
echo “4) 🔧 Setup Environment”
echo “”
read -p “Enter your choice (1-4): “ choice

case $choice in
    1)
        echo “🌐 Opening HTML demo…”
        if command -v xdg-open > /dev/null; then
            xdg-open interactive_demo.html
        elif command -v open > /dev/null; then
            open interactive_demo.html
        else
            echo “Please open interactive_demo.html in your browser”
        fi
        ;;
    2)
        echo “📊 Starting Streamlit app…”
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        streamlit run streamlit_app.py
        ;;
    3)
        echo “📚 Starting Jupyter notebook…”
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
        jupyter notebook notebooks/pattern_recognition_demo.ipynb
        ;;
    4)
        echo “🔧 Running setup…”
        ./setup.sh
        ;;
    *)
        echo “Invalid choice. Please run the script again.”
        ;;
esac
