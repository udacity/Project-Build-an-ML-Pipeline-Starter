#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# DO NOT MODIFY THE SIGNATURE
def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    # ------------------------------------------------------------------
    # 1) Download input artifact (and record that we used it)
    # ------------------------------------------------------------------
    logger.info("Downloading artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # ------------------------------------------------------------------
    # 2) Load data
    # ------------------------------------------------------------------
    logger.info("Reading dataset")
    df = pd.read_csv(artifact_local_path)

    # Basic hygiene: ensure critical columns exist
    required_cols = {"price", "latitude", "longitude", "last_review"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input data: {missing}")

    # Drop obvious bad rows
    df = df.dropna(subset=["price", "latitude", "longitude"]).copy()

    # ------------------------------------------------------------------
    # 3) Cleaning steps (price bounds, dates, NYC bbox)
    # ------------------------------------------------------------------
    logger.info("Filtering price between %.2f and %.2f", args.min_price, args.max_price)
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    logger.info("Filtering by latitude/longitude bounds")
    in_bbox = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[in_bbox].copy()

    # Finalize
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("Cleaned dataset is empty after filtering steps.")

    # ------------------------------------------------------------------
    # 4) Save cleaned file (use the output_artifact name as the local filename)
    # ------------------------------------------------------------------
    output_path = args.output_artifact  # e.g., "clean_sample.csv"
    logger.info("Saving cleaned data to %s", output_path)
    df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # 5) Log the new artifact (with aliases so tests can use :latest and :reference)
    # ------------------------------------------------------------------
    logger.info(
        "Logging cleaned artifact to W&B: name=%s type=%s",
        args.output_artifact,
        args.output_type,
    )
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_path)
    # CRUCIAL for data_check step:
    run.log_artifact(artifact, aliases=["latest", "reference"])

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified input W&B artifact (e.g., 'sample.csv:latest')",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned artifact to log (e.g., 'clean_sample.csv')",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Artifact type for the cleaned data (e.g., 'clean_sample')",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Short description of the cleaned dataset",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price; rows below this are dropped",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price; rows above this are dropped",
        required=True,
    )

    args = parser.parse_args()
    go(args)
