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

@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config: DictConfig) -> None:
    # W&B grouping
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Ensure WANDB entity is available so short artifact names resolve
    entity = config["main"].get("entity")
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    if not os.environ.get("WANDB_ENTITY"):
        raise RuntimeError(
            "WANDB entity is not set. Add `main.entity: <your_wandb_entity>` to "
            "config.yaml or export WANDB_ENTITY in your shell (e.g., `export WANDB_ENTITY=myteam`)."
        )

    steps_param = config["main"]["steps"]
    active_steps = steps_param.split(",") if steps_param != "all" else _STEPS

    # 1) Download raw data (component repo)
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

    # 2) Basic cleaning (local step)
    if "basic_cleaning" in active_steps:
        mlflow.run(
            "src/basic_cleaning",
            entry_point="main",
            env_manager="conda",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with price/geo filters applied",
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
            },
        )

    # 3) Data checks (pytest runs inside the step's conda env)
    if "data_check" in active_steps:
        mlflow.run(
            "src/data_check",
            entry_point="main",
            env_manager="conda",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"],
                "min_rows": config["data_check"]["min_rows"],
                "max_rows": config["data_check"]["max_rows"],
            },
        )

    # 4) Train/Val/Test split (component repo) — NOTE: no val_size arg here
    if "train_val_test_split" in active_steps:
        mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            entry_point="main",
            env_manager="conda",
            parameters={
                "input": "clean_sample.csv:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"],
            },
        )

    # 5) Train Random Forest (local step)
    if "train_random_forest" in active_steps:
        rf_config_path = os.path.abspath("rf_config.json")
        with open(rf_config_path, "w") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        mlflow.run(
            "src/train_random_forest",
            entry_point="main",
            env_manager="conda",
            parameters={
                "rf_config": rf_config_path,
                "train_data": "train.csv:latest",
                "val_data": "val.csv:latest",
                "random_seed": config["modeling"]["random_seed"],
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
            },
        )

    # 6) (Optional) Test promoted model
    if "test_regression_model" in active_steps:
        mlflow.run(
            "src/test_regression_model",
            entry_point="main",
            env_manager="conda",
            parameters={
                "model_export": "model_export:prod",
                "test_artifact": "test.csv:latest",
            },
        )

if __name__ == "__main__":
    go()
