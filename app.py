"""HTTP API for FieldCare crop disease predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from notebooks.predict import (
    PredictionError,
    build_transform,
    find_latest_checkpoint,
    load_checkpoint,
    predict_pil_image,
    select_device,
)
from database import (
    get_history_stats,
    get_scan_image,
    initialize_database,
    list_scans,
    save_scan,
)
from treatments import get_treatment


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
INFERENCE_LOCK = Lock()


def load_predictor(app: FastAPI) -> None:
    requested_device = os.getenv("FIELDCARE_DEVICE", "auto")
    configured_checkpoint = os.getenv("FIELDCARE_CHECKPOINT")
    checkpoint_path = (
        Path(configured_checkpoint).expanduser().resolve()
        if configured_checkpoint
        else find_latest_checkpoint()
    )
    device = select_device(requested_device)
    model, checkpoint, classes = load_checkpoint(checkpoint_path, device)
    app.state.model = model
    app.state.transform = build_transform(checkpoint)
    app.state.classes = classes
    app.state.device = device
    app.state.load_error = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    try:
        load_predictor(app)
    except Exception as exc:  # Keep health checks available when startup fails.
        app.state.load_error = str(exc)
    yield


app = FastAPI(title="FieldCare Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    load_error = getattr(app.state, "load_error", "Model has not been initialized")
    if load_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction model is unavailable: {load_error}",
        )
    return {
        "status": "ready",
        "device": str(app.state.device),
        "class_count": len(app.state.classes),
    }


def run_prediction(image: Image.Image, filename: str) -> dict:
    with INFERENCE_LOCK:
        result = predict_pil_image(
            app.state.model,
            image,
            app.state.transform,
            app.state.classes,
            app.state.device,
            top_k=1,
            image_name=filename,
        )
    best = result["predictions"][0]
    return {
        "class_key": best["class_key"],
        "crop": best["crop"],
        "disease": best["disease"],
        "display_name": best["display_name"],
        "confidence": best["confidence"],
        "confidence_percent": best["confidence_percent"],
        "inference_ms": result["inference_ms"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    load_error = getattr(app.state, "load_error", "Model has not been initialized")
    if load_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable. Check the backend logs.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPG, JPEG, or PNG image.",
        )

    filename = file.filename or "uploaded-image"
    content_type = file.content_type
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 5 MB or smaller.")

    try:
        with Image.open(BytesIO(payload)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(BytesIO(payload)) as uploaded_image:
            if uploaded_image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Upload a JPG, JPEG, or PNG image.",
                )
            image = uploaded_image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    try:
        result = await run_in_threadpool(run_prediction, image, filename)
        history_id = await run_in_threadpool(
            save_scan,
            filename=filename,
            content_type=content_type,
            image=payload,
            prediction=result,
        )
        return {**result, "history_id": history_id}
    except (PredictionError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The image could not be analyzed. Please try again.",
        ) from exc


@app.get("/treatments/{class_key}")
def treatment(class_key: str) -> dict:
    guidance = get_treatment(class_key)
    if guidance is None:
        raise HTTPException(status_code=404, detail="Treatment guidance was not found for this class.")
    return guidance


@app.get("/history")
def history(
    limit: int = Query(default=100, ge=1, le=100),
    category: str = Query(default="all", pattern="^(all|disease|healthy)$"),
    q: str = Query(default="", max_length=100),
) -> dict:
    records = list_scans(limit=limit, category=category, query=q.strip())
    return {"records": records, "count": len(records)}


@app.get("/history/stats")
def history_stats() -> dict:
    return get_history_stats()


@app.get("/history/{scan_id}/image")
def history_image(scan_id: int) -> Response:
    stored_image = get_scan_image(scan_id)
    if stored_image is None:
        raise HTTPException(status_code=404, detail="Scan image was not found.")
    image, content_type = stored_image
    return Response(
        content=image,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
