#!/usr/bin/env bash
# ==============================================================================
# Enterprise E-Commerce Intelligence Engine (E-CIE)
# One-Click Offline Application Launcher
# ==============================================================================

set -e

# Change directory to project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "🚀 Starting E-Commerce Seller Intelligence Engine (E-CIE)..."
echo "============================================================"

# 1. Check or set up a clean host virtual environment
if [ ! -d "host_env" ]; then
    echo "📦 Setting up local environment for your computer (host_env)..."
    python3 -m venv host_env
    echo "⬇️  Installing required packages (one-time setup, takes ~30-45s)..."
    host_env/bin/pip install --upgrade pip
    host_env/bin/pip install -r requirements.txt
    echo "✅ Setup complete!"
fi

# 2. Check PostgreSQL connection
echo "🔍 Checking database connection..."
host_env/bin/python -c "
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
try:
    conn = psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB', 'ecie_project_db'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'tanvipostgres'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432')
    )
    print('   🟢 PostgreSQL Database: Connected & Ready!')
    conn.close()
except Exception as e:
    print('   🟡 Warning: Database connection check:', e)
"

# 3. Launch Streamlit Web Application
echo ""
echo "============================================================"
echo "🌐 Launching Web App on: http://localhost:8501"
echo "   Press Ctrl + C in this terminal to stop the app."
echo "============================================================"
echo ""

host_env/bin/streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
