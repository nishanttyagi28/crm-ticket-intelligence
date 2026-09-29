"""Score a complaint narrative against the saved routing model."""

from __future__ import annotations

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ticketintel.config import MODEL_PATH
from ticketintel.features import request_frame

app = FastAPI(
    title="CRM Ticket Intelligence",
    version="0.1.0",
    description=(
        "Route a consumer complaint to a product line from the narrative "
        "and a few structured fields."
    ),
)

_bundle = None


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    state: str | None = None
    tags: str | None = None
    date_received: str | None = Field(
        default=None,
        description="Date the complaint was received, YYYY-MM-DD. Optional.",
    )


class PredictResponse(BaseModel):
    product: str
    model: str
    scores: dict[str, float] | None


def get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model file is missing.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health() -> dict:
    if not MODEL_PATH.exists():
        return {"status": "model_missing"}
    bundle = get_bundle()
    return {"status": "ok", "model": bundle["model_name"]}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    bundle = get_bundle()
    frame = request_frame(payload.text, payload.state, payload.tags, payload.date_received)
    builder = bundle["builder"]
    estimator = bundle["estimator"]
    matrix = builder.transform_tree(frame) if bundle["kind"] == "tree" else builder.transform(frame)
    product = str(estimator.predict(matrix)[0])
    scores = None
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(matrix)[0]
        scores = {
            str(label): round(float(value), 6)
            for label, value in zip(estimator.classes_, probabilities, strict=True)
        }
    return PredictResponse(product=product, model=bundle["model_name"], scores=scores)
