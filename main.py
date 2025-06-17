import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]


@hydra.main(config_path=".", config_name='config', version_base="1.3")
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

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
            _ = mlflow.run(
                ".",
                entry_point="basic_cleaning",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Cleaned dataset after removing outliers and fixing date format",
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                ".",
                entry_point="data_check",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split", 'main',
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )

        
        if "train_random_forest" in active_steps:
            rf_config = os.path.join(tmp_dir, "rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp) 
            _ = mlflow.run(
                ".",
                entry_point="train_random_forest",
                env_manager="local",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "test_artifact": "test_data.csv:latest",
                    "rf_config": os.path.abspath(rf_config),
                    "output_artifact": "random_forest_export",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"]
                }
            )


        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                entry_point="main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                }
            )

if __name__ == "__main__":
    go()
