#!/bin/bash

echo "===================================================="
echo "Running postStart command..."
echo "===================================================="

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate nyc_airbnb_dev

# Configure persistent conda auto-activation for user's shell configs
# Check if already configured to prevent duplicates
USER_BASHRC="$HOME/.bashrc"
USER_ZSHRC="$HOME/.zshrc"

if ! grep -q "conda activate nyc_airbnb_dev" "$USER_BASHRC" 2>/dev/null; then
    echo "" >> "$USER_BASHRC"
    echo "# Auto-activate nyc_airbnb_dev conda environment" >> "$USER_BASHRC"
    echo "source /opt/conda/etc/profile.d/conda.sh" >> "$USER_BASHRC"
    echo "conda activate nyc_airbnb_dev" >> "$USER_BASHRC"
    echo "✓ Configured bash to auto-activate nyc_airbnb_dev"
fi

if [ -f "$USER_ZSHRC" ]; then
    if ! grep -q "conda activate nyc_airbnb_dev" "$USER_ZSHRC" 2>/dev/null; then
        echo "" >> "$USER_ZSHRC"
        echo "# Auto-activate nyc_airbnb_dev conda environment" >> "$USER_ZSHRC"
        echo "source /opt/conda/etc/profile.d/conda.sh" >> "$USER_ZSHRC"
        echo "conda activate nyc_airbnb_dev" >> "$USER_ZSHRC"
        echo "✓ Configured zsh to auto-activate nyc_airbnb_dev"
    fi
fi

# Display welcome message only once per session
if [ -z "$CODESPACE_WELCOME_SHOWN" ]; then
    export CODESPACE_WELCOME_SHOWN=1

    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║                                                  ║"
    echo "║     NYC Airbnb ML Pipeline - Codespace          ║"
    echo "║                                                  ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    echo "Environment Information:"
    echo "  Python:  $(python --version 2>&1 | awk '{print $2}')"
    echo "  MLflow:  $(mlflow --version 2>&1 | awk '{print $2}')"
    echo "  Conda:   nyc_airbnb_dev (active)"
    echo "  Disk:    $(df -h /workspaces | tail -1 | awk '{print $4}') available"
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "Quick Start:"
    echo "──────────────────────────────────────────────────"
    echo "  mlflow run .                    # Run full pipeline"
    echo "  mlflow run . -P steps=download  # Run single step"
    echo ""
    echo "  Available steps: download, basic_cleaning,"
    echo "                   data_check, data_split,"
    echo "                   train_random_forest"
    echo ""
    echo "  Note: test_regression_model must be run explicitly"
    echo "        after promoting a model to 'prod' in W&B"
    echo ""
    echo "──────────────────────────────────────────────────"
    echo "Documentation:"
    echo "──────────────────────────────────────────────────"
    echo "  README.md                    # Project overview"
    echo "  CODESPACES_QUICKSTART.md     # Codespaces guide"
    echo ""
    echo "  VS Code Tasks (Ctrl+Shift+P → 'Tasks: Run Task'):"
    echo "    - Run Full Pipeline"
    echo "    - Run individual steps"
    echo "    - Cleanup MLflow Environments"
    echo ""
    echo "══════════════════════════════════════════════════"
    echo ""
fi

echo "postStart command completed successfully!"
