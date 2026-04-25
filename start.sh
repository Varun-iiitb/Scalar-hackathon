#!/bin/bash
set -e

cd /app

echo "=============================================="
echo "  IsoSync — starting training"
echo "  $(date)"
echo "=============================================="

python train.py

echo "=============================================="
echo "  Training complete — launching API server"
echo "  $(date)"
echo "=============================================="

python app.py
