#!/bin/bash

echo "===================================================="
echo "MLflow Conda Environment Cleanup"
echo "===================================================="
echo ""
echo "This script will remove MLflow-created conda environments"
echo "that may have become corrupted or are no longer needed."
echo ""

# List MLflow environments
MLFLOW_ENVS=$(conda info --envs | grep mlflow | cut -f1 -d" " || true)

if [ -z "$MLFLOW_ENVS" ]; then
    echo "No MLflow environments found."
    echo "Nothing to clean up!"
    exit 0
fi

echo "The following MLflow environments will be removed:"
echo "──────────────────────────────────────────────────"
echo "$MLFLOW_ENVS"
echo "──────────────────────────────────────────────────"
echo ""

# Prompt for confirmation
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing MLflow environments..."
echo ""

# Remove each environment
for env in $MLFLOW_ENVS; do
    echo "Removing environment: $env"
    conda uninstall --name "$env" --all -y
done

echo ""
echo "✓ Cleanup completed successfully!"
echo ""
echo "Note: MLflow will recreate these environments as needed"
echo "when you run the pipeline again."
