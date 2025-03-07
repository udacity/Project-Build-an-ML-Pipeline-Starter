import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
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
            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "input_artifact": "lbekel-western-governors-university/nyc_airbnb/sample.csv:latest",
                    "output_artifact": "lbekel-western-governors-university/nyc_airbnb/clean_sample",
                    "output_type": "cleaned_data",
                    "output_description": "Cleaned Airbnb dataset with outlier removal and missing values handled",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "csv": "lbekel-western-governors-university/nyc_airbnb/clean_sample:latest",
                    "ref": "lbekel-western-governors-university/nyc_airbnb/clean_sample:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": "lbekel-western-governors-university/nyc_airbnb/clean_sample:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )


        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "trainval_artifact": "lbekel-western-governors-university/nyc_airbnb/trainval_data:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": "rf_config.json",
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export"
                },
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "mlflow_model": "lbekel-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_train_random_forest/random_forest_export_maxdepth50:prod",
                    "test_dataset": "lbekel-western-governors-university/nyc_airbnb/test_data:latest"
                },
            )


if __name__ == "__main__":
    go()
