import json
import os

import hydra
import mlflow
from omegaconf import DictConfig

_STEPS = [
    "download",
    "basic_cleaning",
    "data_check",
    "train_val_test_split",  # split into train/val/test
    "train_random_forest",
    # "test_regression_model",  # run explicitly when ready
]


# Read configuration from config.yaml at project root
@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config: DictConfig) -> None:
    # Group runs in W&B
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Determine which steps to execute
    steps_param = config["main"]["steps"]
    active_steps = steps_param.split(",") if steps_param != "all" else _STEPS

    # ------------------------------------------------------------------
    # 1) Download raw data (component repo)
    # ------------------------------------------------------------------
    if "download" in active_steps:
        mlflow.run(
            f"{config['main']['components_repository']}/get_data",
            entry_point="main",
            version="main",
            env_manager="conda",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded",
            },
        )

    # ------------------------------------------------------------------
    # 2) Basic cleaning (local step)
    # ------------------------------------------------------------------
    if "basic_cleaning" in active_steps:
        mlflow.run(
            "src/basic_cleaning",
            entry_point="main",
            env_manager="local",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with price/geo filters applied",
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # ------------------------------------------------------------------
    # 3) Data checks (local step — needs pytest from its conda env)
    # ------------------------------------------------------------------
    if "data_check" in active_steps:
        mlflow.run(
            "src/data_check",
            entry_point="main",
            env_manager="conda",  # ensure pytest from conda.yml is available
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # ------------------------------------------------------------------
    # 4) Train/Val/Test split (component repo)
    # ------------------------------------------------------------------
    if "train_val_test_split" in active_steps:
        mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            entry_point="main",
            env_manager="conda",
            parameters={
                "input": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "val_size": config["modeling"]["val_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
        )

    # ------------------------------------------------------------------
    # 5) Train Random Forest (local step)
    # ------------------------------------------------------------------
    if "train_random_forest" in active_steps:
        # Serialize RF config to JSON for the step
        rf_config_path = os.path.abspath("rf_config.json")
        with open(rf_config_path, "w") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        mlflow.run(
            "src/train_random_forest",
            entry_point="main",
            env_manager="local",
            parameters={
                "rf_config": rf_config_path,
                # Match common param names in the step's MLproject
                "train_data": "train.csv:latest",
                "val_data": "val.csv:latest",
            },
        )

    # ------------------------------------------------------------------
    # 6) (Optional) Test promoted model
    # ------------------------------------------------------------------
    if "test_regression_model" in active_steps:
        mlflow.run(
            "src/test_regression_model",
            entry_point="main",
            env_manager="local",
            parameters={
                "model_export": "model_export:prod",
                "test_artifact": "test.csv:latest",
            },
        )


if __name__ == "__main__":
    go()
