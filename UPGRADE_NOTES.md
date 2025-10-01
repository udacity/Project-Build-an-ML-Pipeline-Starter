# Python 3.13 Upgrade Notes

## Summary of Changes

### Environment Updates
1. Updated Python version from 3.10 to 3.13 in all conda.yml files
2. Updated key dependencies:
   - mlflow: 2.8.1 → 2.10.0
   - wandb: 0.16.0 → 0.16.3
   - pandas: 2.1.3 → 2.2.0
   - scikit-learn: 1.5.2 → 1.4.0
   - matplotlib: 3.8.2 → 3.8.3
   - hydra-core: 1.3.2 → 1.3.3
   - ydata-profiling: 4.12.1 → 4.13.0

### Code Updates
1. Updated type hints in test_data.py for better compatibility
2. Modified datetime handling in EDA notebook to use newer pandas features
3. Updated pandas DataFrame operations for compatibility with newer pandas API

All changes have been made with backward compatibility in mind while leveraging the latest features and optimizations available in Python 3.13.
