import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Prometheus metrics ---
total_predictions = Counter(
    "total_predictions",
    "Total number of predictions made",
)
prediction_by_class = Counter(
    "prediction_by_class",
    "Predictions broken down by class",
    ["class_name"],
)
inference_latency = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
prediction_errors = Counter(
    "prediction_errors",
    "Total number of prediction errors",
)
active_predictions = Gauge(
    "active_predictions",
    "Number of predictions currently being processed",
)

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.get("/api")
def root():
    return {
        "name": "Pneumonia CNN API",
        "version": "1.0.0",
        "description": "Clasificación de radiografías de tórax en Normal, Neumonía Bacteriana y Neumonía Vírica",
        "endpoints": {
            "GET /health": "Estado del servicio y del modelo",
            "GET /info": "Metadatos del modelo",
            "POST /predict": "Realiza una predicción sobre una imagen",
            "GET /metrics": "Métricas en formato Prometheus",
        },
    }


@router.get("/health")
def health(request: Request):
    wrapper = request.app.state.wrapper
    return {
        "status": "ok",
        "model_loaded": wrapper is not None and wrapper.is_loaded,
    }


@router.get("/info")
def info(request: Request):
    from app.model.architecture import CLASS_LABELS, CLASS_NAMES

    wrapper = request.app.state.wrapper
    if wrapper is None or not wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {
        "model_name": "PneumoniaCNN",
        "architecture": "3× Conv2d + MaxPool2d + Dropout(0.5) + 2× Linear",
        "input_size": "100×100 px (RGB)",
        "classes": [{"key": name, "label": CLASS_LABELS[name]} for name in CLASS_NAMES],
        "optimizer": "RMSprop",
        "version": "1.0.0",
        "load_timestamp": wrapper.load_timestamp,
    }


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    wrapper = request.app.state.wrapper
    if wrapper is None or not wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de archivo no permitido: '{file.content_type}'. Usa JPG o PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="El archivo está vacío.")
    if len(image_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="El archivo supera el tamaño máximo de 10 MB.")

    active_predictions.inc()
    start = time.perf_counter()

    try:
        result = wrapper.predict(image_bytes)
    except ValueError as exc:
        prediction_errors.inc()
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        prediction_errors.inc()
        logger.error(f"Error en inferencia: {exc}")
        raise HTTPException(status_code=500, detail="Error interno durante la inferencia.")
    finally:
        elapsed = time.perf_counter() - start
        inference_latency.observe(elapsed)
        active_predictions.dec()

    total_predictions.inc()
    prediction_by_class.labels(class_name=result["prediction"]).inc()

    logger.info(
        f"Predicción: {result['prediction']} "
        f"(confianza: {result['confidence']:.2%}, {result['inference_time_ms']:.1f} ms)"
    )

    return result


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


@router.get("/test-images")
def list_test_images():
    from app.model.architecture import CLASS_LABELS

    test_path = os.getenv("TEST_IMAGES_PATH", "")
    if not test_path or not os.path.isdir(test_path):
        raise HTTPException(status_code=404, detail="Carpeta de test no disponible.")

    images = []

    normal_dir = os.path.join(test_path, "NORMAL")
    if os.path.isdir(normal_dir):
        for fname in sorted(os.listdir(normal_dir)):
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                images.append({
                    "filename": fname,
                    "class": "NORMAL",
                    "label": CLASS_LABELS["NORMAL"],
                    "url": f"/test-static/NORMAL/{fname}",
                })

    pneumonia_dir = os.path.join(test_path, "PNEUMONIA")
    if os.path.isdir(pneumonia_dir):
        for fname in sorted(os.listdir(pneumonia_dir)):
            if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                continue
            name_lower = fname.lower()
            if "bacteria" in name_lower:
                cls = "PNEUMONIA_BACTERIAL"
            elif "virus" in name_lower:
                cls = "PNEUMONIA_VIRAL"
            else:
                continue
            images.append({
                "filename": fname,
                "class": cls,
                "label": CLASS_LABELS[cls],
                "url": f"/test-static/PNEUMONIA/{fname}",
            })

    by_class = {}
    for img in images:
        by_class[img["class"]] = by_class.get(img["class"], 0) + 1

    return {"images": images, "total": len(images), "by_class": by_class}
