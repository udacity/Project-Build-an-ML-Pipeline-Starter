import os
import json
import tempfile
import mlflow
from omegaconf import DictConfig
import wandb
import hydra
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # Uncomment to run this step if needed
    # "test_regression_model"
]

@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # Step: Basic Cleaning
            # Cleans raw data by removing outliers and handling nulls
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "cleaned_data.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Data after basic cleaning",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            # Step: Data Check
            # Validate cleaned data quality
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "csv": "cleaned_data.csv:latest",
                    "ref": "sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            # Step: Data Split
            # Split cleaned data into train, test, and validation sets
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_split",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input": "cleaned_data.csv:latest",
                    "artifact_root": "data",
                    "artifact_type": "split_data",
                    "test_size": config["modeling"]["test_size"],
                    "val_size": config["modeling"]["val_size"]
                },
            )

        if "train_random_forest" in active_steps:
            # Step: Train Random Forest
            # Train a random forest regressor
            rf_config = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_random_forest",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "train_data": "data_train.csv:latest",
                    "val_data": "data_val.csv:latest",
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "artifact_root": "random_forest_export",
                    "artifact_type": "model_export",
                    "artifact_description": "Random Forest Model Export"
                },
            )

        if "test_regression_model" in active_steps:
            # Step: Test Regression Model
            # Test the trained model on the test set
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "model_export": "random_forest_export:prod",
                    "test_data": "data_test.csv:latest"
                },
            )

if __name__ == "__main__":
    go()
