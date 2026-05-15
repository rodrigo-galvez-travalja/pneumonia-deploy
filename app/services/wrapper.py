"""
Wrapper de inferencia para la CNN de diagnóstico de neumonía.

Encapsula toda la lógica de carga del modelo, preprocesamiento,
inferencia y postprocesamiento. Completamente desacoplado de FastAPI.
"""

import time
import logging
from datetime import datetime, timezone

import torch

from app.model.architecture import PneumoniaCNN, CLASS_NAMES, CLASS_LABELS
from app.utils.preprocessing import preprocess_image

logger = logging.getLogger(__name__)


class PneumoniaWrapper:
    """
    Wrapper de inferencia para la CNN de neumonía.

    Responsabilidades:
        - Cargar la arquitectura y los pesos del modelo
        - Preprocesar imágenes de entrada
        - Ejecutar inferencia de forma determinista
        - Postprocesar y formatear resultados

    Args:
        model_path (str): Ruta al archivo de pesos (.pth)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.load_timestamp: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Carga la arquitectura y los pesos del modelo desde disco.

        Raises:
            RuntimeError: Si el archivo de pesos no se puede cargar
        """
        try:
            logger.info(f"Cargando modelo desde {self.model_path}...")

            self.model = PneumoniaCNN(num_classes=len(CLASS_NAMES))

            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))

            # Soportar tanto state_dict directo como checkpoint con metadatos
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            # Modo evaluación: desactiva dropout y batch norm para inferencia
            self.model.eval()
            self.is_loaded = True
            self.load_timestamp = datetime.now(timezone.utc).isoformat()

            logger.info("Modelo cargado correctamente.")

        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo: {e}")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Realiza la inferencia sobre una imagen.

        Args:
            image_bytes: Contenido binario de la imagen (jpg, jpeg, png)

        Returns:
            dict con los campos:
                - prediction (str): Clase con mayor probabilidad
                - confidence (float): Probabilidad de la clase predicha
                - inference_time_ms (float): Tiempo de inferencia en ms
                - probabilities (list): Lista ordenada de clases y probabilidades

        Raises:
            RuntimeError: Si el modelo no está cargado
            ValueError: Si la imagen no es válida
        """
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado.")

        # Preprocesar imagen
        tensor = preprocess_image(image_bytes)

        # Inferencia determinista sin cálculo de gradientes
        start_time = time.time()

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        # Postprocesar resultados
        return self._postprocess(probabilities, inference_time_ms)

    def _postprocess(self, probabilities: torch.Tensor, inference_time_ms: float) -> dict:
        """
        Formatea la salida del modelo en un dict JSON-serializable.

        Args:
            probabilities: Tensor de probabilidades con shape (num_classes,)
            inference_time_ms: Tiempo de inferencia medido

        Returns:
            dict con predicción, confianza, tiempo y probabilidades ordenadas
        """
        # Convertir a lista de Python (tipos JSON serializables)
        probs_list = probabilities.tolist()

        # Construir lista de clases con sus probabilidades
        class_probs = [
            {
                "class": CLASS_NAMES[i],
                "label": CLASS_LABELS[CLASS_NAMES[i]],
                "probability": round(probs_list[i], 4)
            }
            for i in range(len(CLASS_NAMES))
        ]

        # Ordenar de mayor a menor probabilidad
        class_probs.sort(key=lambda x: x["probability"], reverse=True)

        top_class = class_probs[0]

        return {
            "prediction": top_class["class"],
            "label": top_class["label"],
            "confidence": top_class["probability"],
            "inference_time_ms": inference_time_ms,
            "probabilities": class_probs
        }
