#!/bin/bash
set -e

cd /app

echo "=============================================="
echo "  DubGuard — starting training"
echo "  $(date)"
echo "=============================================="

python training/train.py

echo "=============================================="
echo "  Training complete — launching API server"
echo "  $(date)"
echo "=============================================="

python app.py
