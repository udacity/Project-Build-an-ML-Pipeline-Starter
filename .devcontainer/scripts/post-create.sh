#!/bin/bash
set -e

echo "===================================================="
echo "Running postCreate command..."
echo "===================================================="

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate nyc_airbnb_dev

# Verify installations
echo "Verifying Python installation..."
python --version

echo "Verifying conda installation..."
conda --version

echo "Verifying MLflow installation..."
mlflow --version

echo "Verifying wandb installation..."
wandb --version

# Check for WANDB_API_KEY
echo ""
echo "===================================================="
echo "Checking Weights & Biases API Key..."
echo "===================================================="

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY environment variable is not set!"
    echo ""
    echo "To set your W&B API key:"
    echo "1. Go to your GitHub repository Settings"
    echo "2. Navigate to Secrets and variables → Codespaces"
    echo "3. Add a new secret named 'WANDB_API_KEY'"
    echo "4. Get your API key from https://wandb.ai/authorize"
    echo "5. Rebuild your Codespace"
    echo ""
    echo "You can also set it manually by running:"
    echo "  export WANDB_API_KEY='your-key-here'"
    echo ""
else
    echo "✓ WANDB_API_KEY is set"

    # Validate v1 API key format
    if [[ $WANDB_API_KEY == wandb_v1_* ]]; then
        echo "✓ Using new v1 API key format"

        # Attempt auto-login
        echo "Attempting to login to W&B..."
        set +e  # Temporarily allow errors for W&B login
        if echo "$WANDB_API_KEY" | wandb login --relogin 2>/dev/null; then
            echo "✓ Successfully logged in to Weights & Biases"
        else
            echo "⚠️  Warning: wandb login failed. You may need to login manually."
        fi
        set -e  # Re-enable exit on error
    else
        echo "⚠️  Warning: API key does not match new v1 format (wandb_v1_...)"
        echo "   Please update to a new API key from https://wandb.ai/authorize"
        echo "   Old format keys may not work properly."
    fi
fi

echo ""
echo "postCreate command completed successfully!"
