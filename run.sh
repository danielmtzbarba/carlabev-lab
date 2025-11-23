#!/bin/bash
set -e

echo "========================================"
echo "🚀 Running Experiment 1"
echo "========================================"
uv run train.py exp --exp-id 11

echo "========================================"
echo "🚀 Running Experiment 2"
echo "========================================"
uv run train.py exp --exp-id 12

echo "========================================"
echo "🚀 Running Experiment 3"
echo "========================================"
uv run train.py exp --exp-id 15

echo "========================================"
echo "🚀 Running Experiment 3"
echo "========================================"

uv run train.py exp --exp-id 16
echo "========================================"
echo "🎉 All experiments completed!"
echo "========================================"
