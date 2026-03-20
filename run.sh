#!/bin/bash
# ─────────────────────────────────────────────────────
# run.sh  —  Quick launcher for the FEA Tube Analyser
#
# In your Codespaces terminal, just type:
#   bash run.sh
#
# Then choose which app to run.
# ─────────────────────────────────────────────────────

echo ""
echo "  FEA Tube Analyser — Launcher"
echo "  ─────────────────────────────"
echo "  1)  Streamlit app  (recommended — pure Python dashboard)"
echo "  2)  Flask app      (HTML + Python backend)"
echo ""
read -p "  Enter 1 or 2: " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "  Starting Streamlit on port 8501..."
    echo "  Codespaces will show a popup — click 'Open in Browser'"
    echo ""
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

elif [ "$choice" = "2" ]; then
    echo ""
    echo "  Starting Flask on port 8080..."
    echo "  Go to the Ports tab in VS Code and click the link for port 8080"
    echo ""
    python main.py

else
    echo "  Invalid choice. Run the script again and enter 1 or 2."
fi
