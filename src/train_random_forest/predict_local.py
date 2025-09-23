#!/usr/bin/env python
import argparse
from pathlib import Path
import mlflow
import pandas as pd

# Resolve paths relative to this file, so it works no matter the CWD
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "random_forest_dir"
DEFAULT_CSV = HERE.parent / "basic_cleaning" / "clean_sample.csv"

def main(model_path: str, csv_path: str, n_rows: int):
    model_path = str(model_path)
    csv_path = str(csv_path)

    print(f"Using model: {model_path}")
    print(f"Using data : {csv_path}")

    model = mlflow.sklearn.load_model(model_path)
    X = (pd.read_csv(csv_path)
           .drop(columns=["price"], errors="ignore")
           .head(n_rows))
    preds = model.predict(X)
    print("Predictions:", [float(p) for p in preds])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--csv",   default=str(DEFAULT_CSV))
    p.add_argument("--n_rows", type=int, default=3)
    # parse_known_args so IDE-added flags don't crash the script
    args, _ = p.parse_known_args()
    main(args.model, args.csv, args.n_rows)
