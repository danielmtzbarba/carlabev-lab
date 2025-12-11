#!/bin/bash
set -e

for i in {1..24}; do
	echo "========================================"
	echo "🚀 Running Experiment $i"
	echo "========================================"
	uv run eval.py exp --exp-id $i
done
