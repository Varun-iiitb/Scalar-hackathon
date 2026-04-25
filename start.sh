#!/bin/bash
set -e

cd /app

echo "=============================================="
echo "  IsoSync — starting API server"
echo "  $(date)"
echo "=============================================="

# Start server immediately so HF Spaces health check passes
python app.py &
SERVER_PID=$!

# Give it a moment to bind the port
sleep 3

echo "=============================================="
echo "  IsoSync — starting training"
echo "  $(date)"
echo "=============================================="

python train.py

echo "=============================================="
echo "  Training complete — server still running"
echo "  $(date)"
echo "=============================================="

# Keep container alive while server runs
wait $SERVER_PID
