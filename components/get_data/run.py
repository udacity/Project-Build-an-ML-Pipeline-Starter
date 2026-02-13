#!/usr/bin/env python
"""
This script downloads a URL to a local destination
"""
import argparse
import logging
import os
import glob
import re
import wandb

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def sanitize_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def go(args):

    run = wandb.init(job_type="download_file")
    run.config.update(vars(args))

    if ":" in args.sample:
        #W&B artifact reference
        artifact_ref = run.use_artifact(args.sample)
        artifact_local_path = artifact_ref.download()
        files = glob.glob(os.path.join(artifact_local_path, "*"))
        if not files:
            raise FileNotFoundError(f"No files found in artifact {args.sample}")
        local_file_path = files[0]
    else:
        #Local file case
        local_file_path = os.path.join("data", args.sample)
        if not os.path.isfile(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

    logger.info(f"Returning sample {args.sample}")
    

    #Ensure clean artifact name
    safe_artifact_name = sanitize_artifact_name(args.artifact_name)
    logger.info(f"Logging artifact as: {safe_artifact_name}")

    #log the artifact
    artifact = wandb.Artifact(
        name=safe_artifact_name,
        type=args.artifact_type,
        description=args.artifact_description
    )
    artifact.add_file(local_file_path)
    run.log_artifact(artifact)
    logger.info("Artifact logged successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)
