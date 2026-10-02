#!/bin/bash
# Trains a model for every configuration in configs/ and evaluates it with the given derivation threshold.
# Usage: ./run_all.sh [threshold]
set -o pipefail
if [ $# -ne 1 ]; then
    echo "Usage: $0 [threshold]" >&2
    exit 1
fi
for config in ./configs/*.yaml; do
    echo "Training with $config"
    # train prints the new model folder last
    model=$(python -m src.run.train "$config" | tee /dev/stderr | tail -n 1) || { echo "Training failed for $config"; continue; }
    python -m src.run.evaluate --model "$model" --threshold "$1"
done
