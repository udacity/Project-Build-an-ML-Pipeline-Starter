import logging
import os
import shutil
import matplotlib.pyplot as plt
import mlflow
import json 
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

import hydra
from omegaconf import DictConfig, OmegaConf


def delta_date_feature(dates):
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def go(config: DictConfig):

    run = wandb.init(
        job_type="train_random_forest",
        project=config.main.project_name,
        config=OmegaConf.to_container(config, resolve=True)
    )

    logger.info(f"Loading Random Forest config from: {config.rf_config}")
    with open(config.rf_config, 'r') as fp:
        rf_config = json.load(fp)
        
    rf_config.pop("rf_config", None)
    
    rf_config["random_state"] = int(config.random_seed)
    
    run.config.update(rf_config)
                
    logger.info(f"Downloading trainval_artifact: {config.trainval_artifact}")
    trainval_data_file = run.use_artifact(config.trainval_artifact).file()
    X = pd.read_csv(trainval_data_file)
    y = X.pop("price")

    logger.info(f"Downloading test_artifact: {config.test_artifact}")
    test_data_file = run.use_artifact(config.test_artifact).file()
    X_test = pd.read_csv(test_data_file)
    y_test = X_test.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    stratify_data = X[config.stratify_by] if config.stratify_by and config.stratify_by != "none" else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.val_size,
        stratify=stratify_data,
        random_state=config.random_seed 
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config, config.max_tfidf_features)

    logger.info("Fitting")
    sk_pipe.fit(X_train, y_train)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Validation R2: {r_squared:.2f}")
    logger.info(f"Validation MAE: {mae:.2f}")

    logger.info("Exporting model")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        input_example = X_train.iloc[:5]
    )

    artifact = wandb.Artifact(
        config.output_artifact,
        type = 'model_export',
        description = 'Trained random forest artifact',
        metadata = rf_config
    )
    artifact.add_dir('random_forest_dir')
    run.log_artifact(artifact)

    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    run.summary['r2'] = r_squared
    run.summary['mae'] = mae

    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )
    wandb.finish()


def plot_feature_importance(pipe, feat_names):
    feat_imp = pipe.named_steps["random_forest"].feature_importances_[: len(feat_names)-1]
    nlp_importance = sum(pipe.named_steps["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown='ignore')
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    random_forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(
        steps =[
        ('preprocessor', preprocessor),
        ('random_forest', random_forest)
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":
    go()
