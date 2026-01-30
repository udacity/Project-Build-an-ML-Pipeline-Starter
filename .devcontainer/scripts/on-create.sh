#!/bin/bash
set -e

echo "===================================================="
echo "Running onCreate command..."
echo "===================================================="

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate nyc_airbnb_dev

# Install wandb_utils package in editable mode
echo "Installing wandb_utils package..."
cd /workspaces/Project-Build-an-ML-Pipeline-Starter/components
pip install -e .
cd /workspaces/Project-Build-an-ML-Pipeline-Starter

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p mlruns
mkdir -p outputs

# Check for pre-existing sample data
if [ -f "data/sample1.csv" ]; then
    echo "Found existing sample data at data/sample1.csv"
else
    echo "No pre-existing sample data found. Will download on first pipeline run."
fi

echo "onCreate command completed successfully!"
