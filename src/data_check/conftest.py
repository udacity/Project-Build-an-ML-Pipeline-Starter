import pytest
import pandas as pd
import wandb
import os

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store", type=float, default=0.2)
    parser.addoption("--min_price", action="store", type=float, default=10.0)
    parser.addoption("--max_price", action="store", type=float, default=350.0)


@pytest.fixture(scope='session')
def data(request):
    csv_artifact_name = request.config.getoption("--csv")
    
    if not csv_artifact_name:
        pytest.fail("You must provide the --csv option on the command line")

    run = wandb.init(job_type="data_tests", resume="allow")
    
    try:
        current_artifact = run.use_artifact(csv_artifact_name)
        data_dir = current_artifact.download()
        df = pd.read_csv(os.path.join(data_dir, "clean_sample.csv"))
    except Exception as e:
        pytest.fail(f"Failed to load data from artifact '{csv_artifact_name}': {e}")
    finally:
        run.finish()
        
    return df


@pytest.fixture(scope='session')
def ref_data(request):
    
    ref_artifact_name = request.config.getoption("--ref")
    
    if not ref_artifact_name:
        pytest.fail("You must provide the --ref option on the command line")

    run = wandb.init(job_type="data_tests", resume="allow")
    
    try:
        ref_artifact = run.use_artifact(ref_artifact_name)
        ref_data_dir = ref_artifact.download()
        df = pd.read_csv(os.path.join(ref_data_dir, "clean_sample.csv"))
    except Exception as e:
        pytest.fail(f"Failed to load reference data from artifact '{ref_artifact_name}' : {e}")
    finally:
        run.finish()
    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    return request.config.getoption("--kl_threshold")

@pytest.fixture(scope='session')
def min_price(request):
    return request.config.getoption("--min_price")

@pytest.fixture(scope='session')
def max_price(request):
    return request.config.getoption("--max_price")