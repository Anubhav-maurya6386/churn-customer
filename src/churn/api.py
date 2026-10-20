"""REST API.   uvicorn churn.api:app --port 8000    (docs at /docs)"""
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import __version__
from .config import get_logger
from .predict import ChurnModel, InferenceError

log = get_logger("churn.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app):
    try:
        state["model"] = ChurnModel.load()
        log.info("model loaded: %s v%s", state["model"].meta["model_family"], state["model"].meta["version"])
    except FileNotFoundError as e:
        log.warning(str(e))
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version=__version__, lifespan=lifespan)


class PredictRequest(BaseModel):
    customers: list[dict] = Field(min_length=1, max_length=5000,
                                  description="Customer records with the fields listed in GET /model-info")
    explain: bool = True


def _model() -> ChurnModel:
    if "model" not in state:
        raise HTTPException(503, "Model not loaded. Train it first: python -m churn.train")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok" if "model" in state else "model_not_loaded", "api_version": __version__}


@app.get("/model-info")
def model_info():
    m = _model().meta
    return {"model_family": m["model_family"], "version": m["version"], "trained_at": m["trained_at"],
            "required_fields": m["raw_features"], "categorical_levels": m["category_levels"],
            "test_metrics": m["metrics"], "business_assumptions": m["business"]}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.customers)
    try:
        res = _model().score(df, explain=req.explain)
    except InferenceError as e:
        raise HTTPException(422, str(e))
    return {"n": len(res), "predictions": res.to_dict(orient="records")}
