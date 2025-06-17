import argparse
import logging
import wandb
import pandas as pd
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning", project="nyc_airbnb", group="cleaning", save_code=True)
    run.config.update(args)
 
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    
    if artifact.type == "directory":
        artifact_dir = artifact.download()
        input_csv_path = os.path.join(artifact_dir, "sample.csv")
    else:
        input_csv_path = artifact.file()

    logger.info(f"Reading data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_filepath = os.path.join(tmp_dir, "clean_sample.csv")
        df.to_csv(output_filepath, index=False)

        output_artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        output_artifact.add_file(output_filepath) 
        run.log_artifact(output_artifact)
    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type = str,
        help = "Name of the initial artifact to be cleaned",
        required = True
    )

    parser.add_argument(
        "--output_artifact", 
        type = str,
        help = "Name of the cleaned output artifact",
        required = True
    )

    parser.add_argument(
        "--output_type", 
        type = str,
        help = "Type classification of the cleaned artifact",
        required = True
    )

    parser.add_argument(
        "--output_description", 
        type = str,
        help = "Description of the cleaned artifact",
        required = True
    )

    parser.add_argument(
        "--min_price", 
        type = float,
        help = "Minimum price threshold for filtering outliers",
        required = True
    )

    parser.add_argument(
        "--max_price",
        type = float,
        help = "Maximum price threshold for filtering outliers",
        required = True
    )


    args = parser.parse_args()
    go(args)