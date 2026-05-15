# CLAUDE.md — Proyecto: Despliegue CNN Diagnóstico de Neumonía

## Contexto del proyecto

Este proyecto forma parte de la asignatura **Aplicaciones de Inteligencia Artificial II** (curso 24-25).
El objetivo es desplegar de forma profesional una CNN entrenada en el Lab 03 (Práctica 2) para clasificar radiografías de tórax.

**Alumno:** Rodrigo Gálvez Travalja
**Grupo:** Víctor Teruel, Antonio Gallego, Francisco Javier Garrido, Javier Ramos

Sistema completo de despliegue de una CNN médica para clasificación de radiografías de tórax.

**Objetivo:** Clasificar imágenes Chest X-Ray en 3 clases:
- NORMAL
- PNEUMONIA_BACTERIAL
- PNEUMONIA_VIRAL

El sistema debe incluir: entrenamiento PyTorch, wrapper de inferencia, API REST FastAPI, frontend web, métricas Prometheus, dashboard Grafana, Docker, docker-compose y versionado GitHub.

No reutilizar el notebook del taller de corazones. Todo el código debe ser nuevo y específico para neumonía.

---

## Stack

- Python 3.11
- PyTorch + torchvision
- FastAPI + Uvicorn + Pydantic
- Pillow + NumPy
- Prometheus Client
- Grafana + Prometheus
- Docker + Docker Compose
- HTML + CSS + JavaScript (vanilla)

---

## Dataset

- **Nombre:** Chest X-Ray Images (Pneumonia)
- **Fuente:** Kaggle — `paultimothymooney/chest-xray-pneumonia`
- **Clases:** NORMAL · PNEUMONIA_BACTERIAL · PNEUMONIA_VIRAL
- El pipeline debe soportar los splits: train / val / test
- Nunca hardcodear rutas absolutas. Usar variables de entorno o rutas relativas.

---

## Model Requirements

Usar PyTorch exclusivamente. La CNN debe:
- Aceptar imágenes RGB
- Input final 100×100
- Usar softmax final de 3 clases
- Guardar pesos con `torch.save()` y cargarlos con `torch.load()`

**Arquitectura exacta (Lab 03 Práctica 2):**
- Conv2d(3→32, k=3, padding=1) + ReLU + MaxPool2d(2,2)
- Conv2d(32→64, k=3, padding=1) + ReLU + MaxPool2d(2,2)
- Conv2d(64→128, k=3, padding=1) + ReLU + MaxPool2d(2,2)
- Dropout(0.5)
- Linear(128×12×12 → 128) + ReLU
- Linear(128 → 3)

**Optimizador:** RMSprop · **Loss:** CrossEntropyLoss

Guardar: arquitectura + state_dict + metadatos mínimos.
Formato: `models/model.pth`

No usar notebooks para inferencia en producción. Todo debe existir como módulos Python reutilizables.

---

## Preprocessing Rules

Pipeline obligatorio para inferencia:
- Resize a 100×100
- Convertir a RGB
- Transformar a tensor
- Normalización ImageNet: mean=[0.485, 0.456, 0.406] · std=[0.229, 0.224, 0.225]

La inferencia debe ser determinista. Usar `model.eval()`. Desactivar gradientes con `torch.no_grad()`.

---

## Prediction Output

La inferencia debe devolver: clase predicha, confianza principal, probabilidades completas ordenadas descendentemente.

```json
{
  "prediction": "Viral Pneumonia",
  "confidence": 0.93,
  "inference_time_ms": 12.4,
  "probabilities": [
    {"class": "Viral Pneumonia",    "probability": 0.93},
    {"class": "Bacterial Pneumonia","probability": 0.05},
    {"class": "Normal",             "probability": 0.02}
  ]
}
```

Nunca devolver tensores PyTorch en respuestas. Convertir siempre a tipos JSON serializables.

---

## Wrapper Rules

Crear una clase dedicada `PneumoniaWrapper` para inferencia. Responsabilidades:
- Cargar modelo y pesos
- Preprocesar imágenes
- Ejecutar inferencia
- Postprocesar resultados

Separar claramente: training / inference / api. Nunca mezclar lógica FastAPI dentro del wrapper.
El wrapper debe ser reutilizable, desacoplado y fácilmente testeable.

---

## API Requirements

Framework obligatorio: **FastAPI**

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Info general |
| GET | `/health` | Estado del servicio y modelo |
| GET | `/info` | Metadatos del modelo |
| POST | `/predict` | Recibe imagen multipart, devuelve predicción |
| GET | `/metrics` | Métricas en formato Prometheus |

Usar `multipart/form-data` para imágenes. Validar: tipo MIME, imágenes corruptas, archivos vacíos.
Manejar errores con `HTTPException`. Nunca permitir crash del servidor por inputs inválidos.

**GET /health**
```json
{"status": "ok", "model_loaded": true}
```

**GET /info** — debe devolver: nombre del modelo, arquitectura, clases, timestamp de carga, versión.

**POST /predict** — aceptar imagen, ejecutar inferencia, medir tiempo, actualizar métricas Prometheus, devolver JSON consistente. No guardar imágenes subidas salvo necesidad explícita.

---

## Metrics Requirements

Usar `prometheus_client`. Exponer métricas en `/metrics`.

Métricas mínimas:
- `total_predictions` — Counter
- `prediction_by_class` — Counter con label de clase
- `inference_latency_seconds` — Histogram
- `prediction_errors_total` — Counter
- `active_predictions` — Gauge

Medir latencia real de inferencia.

---

## Frontend Requirements

Frontend simple y limpio. Tecnologías: HTML + CSS + JavaScript vanilla (sin frameworks).

Debe permitir:
- Subir imagen (drag & drop o botón)
- Preview de la radiografía
- Llamar a la API `/predict`
- Mostrar probabilidades con barras visuales y porcentajes
- Destacar el diagnóstico principal
- Mostrar estado de carga y errores de API

No usar frameworks frontend pesados.

---

## Grafana Requirements

Grafana conectado a Prometheus. Dashboard mínimo:
- Total de predicciones
- Distribución por clase
- Latencia de inferencia
- Errores
- Predicciones por minuto

Todo debe funcionar automáticamente con `docker-compose up`.

---

## Docker Requirements

Todo el sistema debe ser contenerizado. Servicios mínimos:
- `api` — FastAPI + modelo
- `prometheus` — scraping de métricas
- `grafana` — dashboard

Usar `docker-compose.yml`. La API debe arrancar automáticamente.
Usar imágenes ligeras cuando sea posible (e.g. `python:3.11-slim`).
No ejecutar como root si puede evitarse.

---

## Recommended Project Structure

```
project/
├── app/
│   ├── api/              ← endpoints FastAPI
│   ├── frontend/         ← index.html (single file)
│   ├── model/            ← arquitectura CNN
│   ├── services/         ← wrapper de inferencia
│   ├── utils/            ← utilidades (preprocessing, logging...)
│   └── main.py           ← punto de entrada FastAPI
├── models/
│   └── model.pth         ← pesos entrenados
├── training/             ← notebook + script de entrenamiento
│   └── L3P2-Pneumonia.ipynb
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── dashboard.json
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Coding Rules

- Usar tipado cuando sea razonable
- Separar: configuración / lógica de negocio / endpoints / utilidades
- Mantener funciones pequeñas
- No duplicar lógica
- Usar nombres explícitos, no variables de una sola letra
- No mezclar entrenamiento con inferencia
- No crear dependencias innecesarias

---

## Error Handling

Manejar: modelo no cargado, imágenes inválidas, errores de inferencia, timeouts, archivos vacíos.
Las respuestas de error deben ser JSON. Nunca exponer stack traces al cliente.

---

## Performance Rules

- Cargar modelo una sola vez al iniciar (en `startup` event de FastAPI)
- Nunca recargar pesos por request
- Evitar operaciones bloqueantes innecesarias
- Mantener inferencia eficiente

---

## Security Rules

- Validar uploads (tipo MIME, tamaño máximo)
- Limitar tipos de archivo aceptados (jpg, jpeg, png)
- No ejecutar código arbitrario
- No confiar en nombres de archivo del usuario

---

## Logging

Usar `logging` estructurado (no `print()` en producción). Registrar:
- Arranque del servidor y carga del modelo
- Errores e inputs inválidos
- Tiempos de inferencia
- Requests importantes

---

## Testing

Crear tests mínimos para:
- Endpoint `/health`
- Endpoint `/predict`
- Wrapper de inferencia
- Carga del modelo

---

## GitHub Requirements

Versionar: código fuente, Dockerfile, docker-compose.yml, requirements.txt, pesos del modelo (`model.pth`), configuración Prometheus, dashboards Grafana.

No subir: datasets completos, entornos virtuales, cachés, `__pycache__`, `.DS_Store`.

---

## Environment Variables

Usar variables de entorno para: rutas de modelos, puertos, configuración runtime.
Nunca hardcodear secretos.

---

## Deployment

El sistema debe arrancar con:
```bash
docker-compose up --build
```

Flujo esperado: API disponible → frontend accesible → Prometheus scrapeando → Grafana operativo.

---

## README Requirements

Documentar: instalación, entrenamiento, ejecución, endpoints, Docker, métricas, Grafana, estructura del proyecto.
Incluir ejemplos `curl` para `/predict`.

---

## Referencia: taller de ejemplo

La carpeta `TallerPipeModeling/` contiene el proyecto del taller (modelo de enfermedad cardíaca con Regresión Logística). Sirve **únicamente como referencia de arquitectura**, no hay que modificarlo ni reutilizarlo. Archivos útiles como patrón:
- `heart_model_wrapper.py` → patrón de wrapper a adaptar para PyTorch
- `api_heart.py` → patrón de API FastAPI con Prometheus
- `Dockerfile` → patrón de Dockerfile a adaptar

---

## Objetivo final

El resultado debe parecer un sistema real de ML en producción. Prioridades:
claridad · modularidad · estabilidad · reproducibilidad · mantenibilidad · separación de responsabilidades · despliegue simple
