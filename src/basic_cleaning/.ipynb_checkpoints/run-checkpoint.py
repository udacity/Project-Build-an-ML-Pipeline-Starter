#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

# DO NOT MODIFY
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):
    
    logger.info('Starting wandb run.')
    run = wandb.init(
        project = 'nyc_airbnb',
        group = 'basic_cleaning',
        job_type="basic_cleaning" 
    )
    run.config.update(args)
    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info('Fetching raw dataset.')
    local_path = wandb.use_artifact('sample.csv:latest').file()
    df = pd.read_csv(local_path)
    
    # EDA with arguments passed into the step
    logger.info('Cleaning data.')
    idx = df['price'].between(float(args.min_price), float(args.max_price))
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    # TODO: add code to fix the issue happened when testing the model
    

    # Save the cleaned data
    logger.info('Saving and exporting cleaned data.')
    df.to_csv('clean_sample.csv', index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type = args.output_type,
        description = args.output_description
    )
    artifact.add_file('clean_sample.csv')
    run.log_artifact(artifact)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type = str,
        help = "Initial artifact to be cleaned",
        required = True
    )

    parser.add_argument(
        "--output_artifact", 
        type = str,
        help = "Output artifact for cleaned data",
        required = True
    )

    parser.add_argument(
        "--output_type", 
        type = str,
        help = "Type of the output dataset",
        required = True
    )

    parser.add_argument(
        "--output_description", 
        type = str,
        help = "Description of the output dataset",
        required = True
    )

    parser.add_argument(
        "--min_price", 
        type = float,
        help = "Minimum house price to be considered",
        required = True
    )

    parser.add_argument(
        "--max_price",
        type = float,
        help = "Maximum house price to be considered",
        required = True
    )


    args = parser.parse_args()

    go(args)
