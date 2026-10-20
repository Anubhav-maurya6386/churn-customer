import pytest
from churn.config import load_config
from churn.data import generate_synthetic
from churn.train import run


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def df(cfg):
    return generate_synthetic(1500, seed=1)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, cfg, df):
    """Tiny, fast model used by all tests that need a fitted artifact."""
    d = tmp_path_factory.mktemp("art")
    out = run(cfg, df=df, artifacts_dir=str(d), reports_dir=str(d / "rep"), fast=True, make_reports=False)
    out["path"] = d / "churn_model.joblib"
    return out
