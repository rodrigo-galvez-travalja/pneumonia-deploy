import logging
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


@router.get("/")
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
            "GET /ui": "Interfaz web",
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
