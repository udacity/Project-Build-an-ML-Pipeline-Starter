import os
import glob
import tempfile
import pandas as pd
import pytest

# ---- Robust import of the Weights & Biases Public API ----
# Tries new and old import paths, then top-level wandb.Api as a fallback.
WandbApi = None
try:
    # Newer SDK layout
    from wandb.sdk.public import Api as WandbApi  # type: ignore[attr-defined]
except Exception:
    try:
        # Older public API path
        from wandb.apis.public import Api as WandbApi  # type: ignore[attr-defined]
    except Exception:
        try:
            import wandb as _wandb
            WandbApi = getattr(_wandb, "Api", None)
        except Exception:
            WandbApi = None


def pytest_addoption(parser):
    parser.addoption("--csv", required=True, help="W&B artifact or local path for current sample")
    parser.addoption("--ref", required=True, help="W&B artifact or local path for reference sample")
    parser.addoption("--kl_threshold", type=float, required=True)
    parser.addoption("--min_price", type=float, required=True)
    parser.addoption("--max_price", type=float, required=True)
    parser.addoption("--min_rows", type=int, default=10)
    parser.addoption("--max_rows", type=int, default=500_000)


@pytest.fixture(scope="session")
def csv(request):
    return request.config.getoption("--csv")


@pytest.fixture(scope="session")
def ref(request):
    return request.config.getoption("--ref")


@pytest.fixture(scope="session")
def kl_threshold(request):
    return request.config.getoption("--kl_threshold")


@pytest.fixture(scope="session")
def min_price(request):
    return request.config.getoption("--min_price")


@pytest.fixture(scope="session")
def max_price(request):
    return request.config.getoption("--max_price")


@pytest.fixture(scope="session")
def min_rows(request):
    return request.config.getoption("--min_rows")


@pytest.fixture(scope="session")
def max_rows(request):
    return request.config.getoption("--max_rows")


def _resolve_artifact_or_local(uri: str) -> str:
    """
    Return a local file path for either a local CSV or a W&B artifact URI.

    Supports short names like 'clean_sample.csv:latest' if WANDB_ENTITY and
    WANDB_PROJECT are set in the environment. Fully-qualified names like
    'entity/project/clean_sample.csv:latest' also work.
    """
    # Already a local file?
    if os.path.exists(uri):
        return uri

    # Treat strings containing ':' as potential W&B artifact refs (e.g., 'file.csv:latest')
    if ":" in uri:
        if WandbApi is None:
            raise RuntimeError(
                "Weights & Biases Public API is not available in this environment. "
                "Ensure 'wandb' is installed in this step's conda env."
            )

        # Expand short name with env, or accept fully-qualified name as-is
        if "/" not in uri:
            entity = os.environ.get("WANDB_ENTITY")
            project = os.environ.get("WANDB_PROJECT")
            if not entity or not project:
                raise RuntimeError(
                    "Short artifact name provided but WANDB_ENTITY/WANDB_PROJECT are not set. "
                    "Set them in the environment (or pass a fully qualified artifact name like "
                    "'entity/project/artifact:alias')."
                )
            fq_name = f"{entity}/{project}/{uri}"
        else:
            fq_name = uri

        api = WandbApi()
        art = api.artifact(fq_name)

        tmpdir = tempfile.mkdtemp(prefix="data_check_")
        local_dir = art.download(root=tmpdir)

        # Find a CSV inside the downloaded artifact directory
        csv_files = glob.glob(os.path.join(local_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in downloaded artifact {fq_name} at {local_dir}"
            )
        return csv_files[0]

    # Not a local file and not an artifact reference
    raise FileNotFoundError(f"Not a local file or recognizable artifact URI: {uri}")


@pytest.fixture(scope="session")
def data(csv):
    return pd.read_csv(_resolve_artifact_or_local(csv))


@pytest.fixture(scope="session")
def ref_data(ref):
    return pd.read_csv(_resolve_artifact_or_local(ref))
