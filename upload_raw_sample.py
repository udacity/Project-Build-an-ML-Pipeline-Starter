#!/usr/bin/env python
"""
Upload sample1.csv to W&B as a new artifact.
"""

import wandb
import os

# CONFIG
PROJECT_NAME = "nyc_airbnb"        
EXPERIMENT_NAME = "raw_data_upload"
INPUT_FILE = os.path.join("components", "get_data", "data", "sample1.csv")
ARTIFACT_NAME = "sample1.csv"
ARTIFACT_TYPE = "raw_data"
ARTIFACT_DESCRIPTION = "Raw sample dataset uploaded for pipeline"

# UPLOAD SCRIPT
run = wandb.init(
    project=PROJECT_NAME,
    group=EXPERIMENT_NAME,
    job_type="upload_raw_sample"
)

# Create artifact
artifact = wandb.Artifact(
    name=ARTIFACT_NAME,
    type=ARTIFACT_TYPE,
    description=ARTIFACT_DESCRIPTION
)

# Add file
artifact.add_file(INPUT_FILE)

# Log artifact to W&B
run.log_artifact(artifact)

# Finish the run
run.finish()

print(f"Uploaded {INPUT_FILE} to W&B as artifact '{ARTIFACT_NAME}'")
