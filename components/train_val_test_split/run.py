#!/usr/bin/env python
"""
This script splits the provided dataframe into test and trainval sets.
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def log_artifact(filename, artifact_name, artifact_type, description, run):
    """
    Log an artifact to W&B.
    """
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    try:
        # Download input artifact
        logger.info(f"Fetching artifact {args.input}")
        artifact_local_path = run.use_artifact(args.input).file()
        df = pd.read_csv(artifact_local_path)

        # Splitting the dataset
        logger.info("Splitting trainval and test")
        trainval, test = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
        )

        # Save and log the splits
        for split_df, split_name in zip([trainval, test], ['trainval', 'test']):
            logger.info(f"Uploading {split_name}_data.csv dataset")
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as temp_file:
                split_df.to_csv(temp_file.name, index=False)
                temp_file.close()

                # Log the artifact
                log_artifact(
                    temp_file.name,
                    f"{split_name}_data",
                    "dataset",
                    f"{split_name} split of the dataset",
                    run,
                )

                # Clean up temporary file
                os.remove(temp_file.name)

    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}")
        raise

    finally:
        # Ensure W&B run is closed
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
