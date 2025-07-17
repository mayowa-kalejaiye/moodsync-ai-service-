#!/bin/bash
# Render deployment script for Python version compatibility

echo "=== RENDER DEPLOYMENT SCRIPT ==="
echo "Current Python version:"
python --version

echo "Python executable location:"
which python

echo "Available Python versions:"
ls -la /usr/bin/python* 2>/dev/null || echo "Standard python binaries not found"

echo "Setting up Python 3.10 environment..."
export PYTHONPATH=/opt/render/project/src
export PYTHON_VERSION=3.10.14

echo "Installing dependencies..."
pip install --upgrade pip==21.3.1
pip install setuptools==57.5.0

echo "Installing application requirements..."
if pip install -r requirements.txt; then
    echo "✅ Standard requirements installed successfully"
else
    echo "❌ Standard requirements failed, trying ultra-minimal..."
    pip install -r requirements-ultra-minimal.txt
fi

echo "=== INSTALLATION COMPLETE ==="
echo "Installed packages:"
pip list | grep -E "(fastapi|uvicorn|pydantic|gunicorn)"

echo "Starting application..."
