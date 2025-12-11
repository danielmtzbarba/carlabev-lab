#!/bin/bash
set -e

echo "========================================"
echo "🚀 Running Experiment 13"
echo "========================================"
uv run train.py exp --exp-id 13

echo "========================================"
echo "🚀 Running Experiment 14"
echo "========================================"
uv run train.py exp --exp-id 14

echo "========================================"
echo "🎉 All experiments completed!"
echo "========================================"
