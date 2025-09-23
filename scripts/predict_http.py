#!/usr/bin/env python
import argparse, pandas as pd, requests

def main(csv_path, url, n_rows):
    X = (pd.read_csv(csv_path)
           .drop(columns=["price"], errors="ignore")
           .head(n_rows))
    payload_csv = X.to_csv(index=False)  # avoids NaN JSON issues
    r = requests.post(url, data=payload_csv, headers={"Content-Type": "text/csv"}, timeout=30)
    print("Status:", r.status_code)
    print("Body:", r.text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--n_rows", type=int, default=3)
    p.add_argument("--url", default="http://127.0.0.1:5000/invocations")
    args = p.parse_args()
    main(args.csv, args.url, args.n_rows)
