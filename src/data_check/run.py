import os
import sys
import subprocess
import argparse
import logging
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="data_check", project="nyc_airbnb", group="data_validation", save_code=True)
    run.config.update(args)

    logger.info(f"Input CSV: {args.csv}")
    logger.info(f"Reference CSV: {args.ref}")
    logger.info(f"KL Threshold: {args.kl_threshold}")
    logger.info(f"Min Price: {args.min_price}")
    logger.info(f"Max Price: {args.max_price}")

    pytest_command = [
        sys.executable,
        "-m", "pytest",
        "src/data_check/",      
        "-vv",           
        f"--csv={args.csv}",
        f"--ref={args.ref}",
        f"--kl_threshold={args.kl_threshold}",
        f"--min_price={args.min_price}",
        f"--max_price={args.max_price}"
    ]

    logger.info(f"Running pytest command: {' '.join(pytest_command)}")

    try:
        result = subprocess.run(pytest_command, check=True, capture_output=True, text=True)
        
        logger.info("Pytest stdout:\n" + result.stdout)
        if result.stderr:
            logger.warning("Pytest stderr (warnings/errors):\n" + result.stderr)
            run.log({"pytest_stderr": result.stderr})

        logger.info("All data checks passed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Data checks failed: {e.returncode}")
        logger.error("Pytest stdout:\n" + e.stdout)
        logger.error("Pytest stderr:\n" + e.stderr)
        run.log({"pytest_failure_stdout": e.stdout, "pytest_failure_stderr": e.stderr})
        sys.exit(e.returncode)
    finally:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data checks using pytest via MLflow.")

    parser.add_argument(
        "--csv",
        type=str,
        help="Input CSV file to be tested (W&B artifact path)",
        required=True
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="Reference CSV file to compare the new csv to (W&B artifact path)",
        required=True
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        help="Threshold for the KL divergence test on the neighborhood group column",
        required=True
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum accepted price",
        required=True
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted price",
        required=True
    )

    args = parser.parse_args()
    go(args)