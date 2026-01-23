#!/bin/bash

# ============================================================================
# RGCNFormer Backend Startup Script
# ============================================================================
# This script starts both Gunicorn (Flask) and Celery worker services
# with proper conda environment activation.
# ============================================================================

# ============================================================================
# Conda Environment Configuration
# ============================================================================
CONDA_ENV_NAME="learn"        # Change to your conda environment name
CONDA_BASE_PATH="$HOME/miniconda3"  # Conda installation path

# Activate conda environment
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment '$CONDA_ENV_NAME'"
    exit 1
fi

echo "✓ Conda environment activated: $CONDA_ENV_NAME"

# ============================================================================
# Service Configuration
# ============================================================================
WORKER_COUNT=4
HOST=0.0.0.0
PORT=8000

# Change to backend directory
cd "$(dirname "$0")"

echo "=========================================="
echo "  RGCNFormer Backend Startup"
echo "=========================================="
echo "Conda Environment: $CONDA_ENV_NAME"
echo "Working Directory: $(pwd)"
echo "=========================================="

# ============================================================================
# Start Celery Worker (Background)
# ============================================================================
echo "🚀 Starting Celery worker..."
celery -A tasks.celery_app worker \
    --loglevel=info \
    --pidfile=celery.pid \
    --logfile=celery.log &

CELERY_PID=$!
sleep 2

# Check if Celery started successfully
if ps -p $CELERY_PID > /dev/null; then
    echo "✓ Celery worker started successfully (PID: $CELERY_PID)"
else
    echo "❌ Failed to start Celery worker"
    exit 1
fi

# ============================================================================
# Start Gunicorn Server (Foreground)
# ============================================================================
echo "🚀 Starting Gunicorn server (Workers: $WORKER_COUNT, $HOST:$PORT)..."
gunicorn -w $WORKER_COUNT \
    -b $HOST:$PORT \
    --timeout 120 \
    --access-logfile gunicorn_access.log \
    --error-logfile gunicorn_error.log \
    wsgi:app

# ============================================================================
# Cleanup (When Gunicorn exits)
# ============================================================================
echo ""
echo "Gunicorn stopped, cleaning up Celery worker..."
if ps -p $CELERY_PID > /dev/null; then
    kill $CELERY_PID
    echo "✓ Celery worker stopped"
else
    echo "⚠ Celery worker not running"
fi

echo "Backend services stopped completely"
