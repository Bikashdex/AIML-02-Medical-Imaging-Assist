"""FastAPI inference service. Boots without weights so the container is demonstrable early."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.data.dicom_io import describe, read_study, to_rgb_uint8

DISCLAIMER = "For education only. Not for clinical use."
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/classifier.pt")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")

app = FastAPI(
    title="AIML-02 Medical Imaging Inference",
    version=MODEL_VERSION,
    description=f"{DISCLAIMER}",
)

_bundle: dict[str, object] | None = None


def load_bundle() -> dict[str, object] | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = checkpoint["model"].to(device).eval()
    _bundle = {"model": model, "threshold": float(checkpoint.get("threshold", 0.5)), "device": device}
    return _bundle


class Prediction(BaseModel):
    study_uid: str
    prediction: str
    probability: float = Field(ge=0.0, le=1.0)
    threshold_used: float
    model_version: str
    inference_ms: float
    dicom_tags: dict[str, object] | None = None
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health() -> dict[str, object]:
    bundle = load_bundle()
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "device": bundle["device"] if bundle else None,
        "disclaimer": DISCLAIMER,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model": "densenet121",
        "version": MODEL_VERSION,
        "task": "binary lung-opacity detection on chest X-ray",
        "label_semantics": "1 = lung opacity present; 0 = normal OR another abnormality",
        "split_policy": "grouped split on patient_id (one image per group in RSNA stage_2)",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), study_uid: str | None = None) -> Prediction:
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"no model at {MODEL_PATH}; train and export first")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    tmp = Path("/tmp") / (file.filename or "study.dcm")
    tmp.write_bytes(raw)

    tags: dict[str, object] | None = None
    uid = study_uid or (file.filename or "unknown")
    try:
        ds = read_study(tmp)
        tags = describe(ds)
        uid = study_uid or str(tags.get("study_instance_uid") or tags.get("patient_id") or uid)
        array = to_rgb_uint8(ds)
    except Exception:
        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(raw)) as img:
            array = np.asarray(img.convert("RGB"))

    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    tensor = tensor.unsqueeze(0).to(bundle["device"])  # type: ignore[union-attr]

    started = time.perf_counter()
    with torch.no_grad():
        logit = bundle["model"](tensor)  # type: ignore[operator]
        probability = float(torch.sigmoid(logit).item())
    elapsed_ms = (time.perf_counter() - started) * 1000

    threshold = float(bundle["threshold"])  # type: ignore[arg-type]
    return Prediction(
        study_uid=uid,
        prediction="opacity" if probability >= threshold else "no_opacity",
        probability=round(probability, 4),
        threshold_used=threshold,
        model_version=MODEL_VERSION,
        inference_ms=round(elapsed_ms, 2),
        dicom_tags=tags,
    )
