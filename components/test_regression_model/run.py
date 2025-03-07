#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

# from wandb_utils.log_artifact import log_artifact



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline

    :param artifact_name: name for the artifact
    :param artifact_type: type for the artifact (just a string like "raw_data", "clean_data" and so on)
    :param artifact_description: a brief description of the artifact
    :param filename: local filename for the artifact
    :param wandb_run: current Weights & Biases run
    :return: None
    """
    # Log to W&B
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()

def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # model_local_path = run.use_artifact(args.mlflow_model).download()
    model_local_path = run.use_artifact("lbekel-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_train_random_forest/random_forest_export_maxdepth50:v0").download()

    # Download test dataset
    # test_dataset_path = run.use_artifact(args.test_dataset).file()
    test_dataset_path = run.use_artifact("lbekel-western-governors-university/nyc_airbnb/test_data.csv:latest").file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    # Log MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the test results artifact",
        required=False
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the test results artifact",
        required=False
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the test results artifact",
        required=False
    )


    args = parser.parse_args()

    go(args)
