import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.services.wrapper import PneumoniaWrapper
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pth")
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando API de diagnóstico de neumonía...")
    try:
        app.state.wrapper = PneumoniaWrapper(MODEL_PATH)
        logger.info("Modelo cargado. API lista.")
    except Exception as exc:
        logger.error(f"No se pudo cargar el modelo: {exc}")
        app.state.wrapper = None
    yield
    logger.info("Apagando API...")


app = FastAPI(
    title="Pneumonia CNN API",
    description="Clasificación de radiografías de tórax mediante CNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/ui", include_in_schema=False)
def frontend():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
