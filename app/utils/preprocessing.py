"""
Pipeline de preprocesamiento de imágenes para inferencia.

Aplica las mismas transformaciones usadas durante el entrenamiento
para garantizar consistencia y resultados deterministas.
"""

import torch
from torchvision import transforms
from PIL import Image
import io


# Tamaño de entrada esperado por la CNN
IMAGE_SIZE = (100, 100)

# Valores de normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pipeline de transformación para inferencia
inference_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesa una imagen en bytes para pasarla a la CNN.

    Pasos:
        1. Decodifica los bytes como imagen PIL
        2. Convierte a RGB (maneja imágenes en escala de grises o RGBA)
        3. Redimensiona a 100x100
        4. Convierte a tensor
        5. Normaliza con valores ImageNet

    Args:
        image_bytes: Contenido binario de la imagen (jpg, jpeg, png)

    Returns:
        Tensor con shape (1, 3, 100, 100) listo para la CNN

    Raises:
        ValueError: Si los bytes no corresponden a una imagen válida
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("No se pudo decodificar la imagen. Asegúrate de subir un archivo JPG o PNG válido.")

    # Convertir siempre a RGB para garantizar 3 canales
    image = image.convert("RGB")

    # Aplicar pipeline de transformación
    tensor = inference_transform(image)

    # Añadir dimensión de batch: (3, 100, 100) → (1, 3, 100, 100)
    tensor = tensor.unsqueeze(0)

    return tensor
