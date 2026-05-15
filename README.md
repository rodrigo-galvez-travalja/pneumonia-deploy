# Pneumonia CNN — Sistema de Diagnóstico Radiológico

Sistema de despliegue profesional de una CNN para clasificación de radiografías de tórax. Detecta tres clases diagnósticas: **Normal**, **Neumonía Bacteriana** y **Neumonía Vírica**.

Proyecto de la asignatura **Aplicaciones de Inteligencia Artificial II** (Curso 24-25).

---

## Índice

1. [Descripción](#descripción)
2. [Arquitectura del sistema](#arquitectura-del-sistema)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Requisitos](#requisitos)
5. [Instalación y ejecución](#instalación-y-ejecución)
6. [Entrenamiento del modelo](#entrenamiento-del-modelo)
7. [Endpoints de la API](#endpoints-de-la-api)
8. [Métricas Prometheus](#métricas-prometheus)
9. [Dashboard Grafana](#dashboard-grafana)
10. [Tests](#tests)

---

## Descripción

La plataforma analiza imágenes de radiografía de tórax y las clasifica en:

| Clase | Descripción |
|---|---|
| `NORMAL` | Pulmones sin patología |
| `PNEUMONIA_BACTERIAL` | Neumonía de origen bacteriano |
| `PNEUMONIA_VIRAL` | Neumonía de origen vírico |

El sistema está basado en una CNN entrenada con PyTorch sobre el dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) de Kaggle.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────┐
│                   docker-compose                    │
│                                                     │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐  │
│  │  FastAPI │──▶│ Prometheus │──▶│   Grafana    │  │
│  │  :8000   │   │   :9090    │   │   :3000      │  │
│  └──────────┘   └────────────┘   └──────────────┘  │
│       │                                             │
│  ┌──────────┐   ┌────────────┐                      │
│  │  Node    │   │  cAdvisor  │                      │
│  │ Exporter │   │   :8081    │                      │
│  │  :9100   │   └────────────┘                      │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

### Stack tecnológico

- **ML:** Python 3.11 · PyTorch · torchvision
- **API:** FastAPI · Uvicorn · Pydantic
- **Preprocesado:** Pillow · NumPy
- **Métricas:** prometheus-client
- **Frontend:** HTML + CSS + JavaScript vanilla
- **Observabilidad:** Prometheus · Grafana · Node Exporter · cAdvisor
- **Contenerización:** Docker · Docker Compose

---

## Estructura del proyecto

```
pneuomnia-deploy/
├── app/
│   ├── api/
│   │   └── endpoints.py        # Endpoints FastAPI
│   ├── frontend/
│   │   └── index.html          # Frontend (single-file, vanilla)
│   ├── model/
│   │   └── architecture.py     # Definición CNN PneumoniaCNN
│   ├── services/
│   │   └── wrapper.py          # PneumoniaWrapper: carga + inferencia
│   ├── utils/
│   │   └── preprocessing.py    # Pipeline de preprocesado de imágenes
│   └── main.py                 # Punto de entrada FastAPI
├── models/
│   └── model.pth               # Pesos entrenados
├── training/
│   └── L3P2-Pneumonia.ipynb    # Notebook de entrenamiento
├── prometheus/
│   └── prometheus.yml          # Configuración scraping
├── grafana/
│   ├── dashboard.json          # Dashboard principal
│   ├── dashboards/             # Provisioning dashboards
│   └── provisioning/           # Provisioning datasources
├── tests/
│   └── test_api.py             # Tests de integración
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Requisitos

- Docker ≥ 24.0
- Docker Compose ≥ 2.0

> No se necesita Python local para ejecutar el sistema completo. Solo es necesario para entrenamiento o desarrollo sin Docker.

---

## Instalación y ejecución

### Levantar el sistema completo

```bash
git clone <url-del-repo>
cd pneuomnia-deploy
docker-compose up --build
```

Servicios disponibles tras el arranque:

| Servicio | URL |
|---|---|
| Frontend + API | http://localhost:8000 |
| Documentación API (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| cAdvisor | http://localhost:8081 |

### Credenciales Grafana

```
Usuario: admin
Contraseña: admin
```

### Parar el sistema

```bash
docker-compose down
```

### Reconstruir tras cambios de código

```bash
docker-compose up --build -d api
```

---

## Entrenamiento del modelo

El notebook de entrenamiento se encuentra en `training/L3P2-Pneumonia.ipynb`.

### Arquitectura CNN

```
Conv2d(3→32, k=3, padding=1) + ReLU + MaxPool2d(2,2)
Conv2d(32→64, k=3, padding=1) + ReLU + MaxPool2d(2,2)
Conv2d(64→128, k=3, padding=1) + ReLU + MaxPool2d(2,2)
Dropout(0.5)
Linear(128×12×12 → 128) + ReLU
Linear(128 → 3)
```

- **Input:** imágenes RGB redimensionadas a 100×100 px
- **Output:** softmax sobre 3 clases
- **Optimizador:** RMSprop
- **Loss:** CrossEntropyLoss
- **Normalización:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Los pesos entrenados se guardan en `models/model.pth`.

---

## Endpoints de la API

### GET `/`

Sirve el frontend web (HTML).

---

### GET `/health`

Estado del servicio y del modelo.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### GET `/info`

Metadatos del modelo cargado.

```bash
curl http://localhost:8000/info
```

```json
{
  "model_name": "PneumoniaCNN",
  "architecture": "CNN",
  "input_size": "100x100 px (RGB)",
  "optimizer": "RMSprop",
  "classes": ["NORMAL", "PNEUMONIA_BACTERIAL", "PNEUMONIA_VIRAL"],
  "load_timestamp": "2025-05-15T10:30:00",
  "version": "1.0.0"
}
```

---

### POST `/predict`

Clasifica una radiografía de tórax. Acepta `multipart/form-data`.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/ruta/a/radiografia.jpg"
```

**Respuesta:**

```json
{
  "prediction": "PNEUMONIA_BACTERIAL",
  "label": "Neumonía Bacteriana",
  "confidence": 0.819,
  "inference_time_ms": 12.4,
  "probabilities": [
    { "class": "PNEUMONIA_BACTERIAL", "label": "Neumonía Bacteriana", "probability": 0.819 },
    { "class": "PNEUMONIA_VIRAL",     "label": "Neumonía Vírica",     "probability": 0.142 },
    { "class": "NORMAL",             "label": "Normal",               "probability": 0.039 }
  ]
}
```

**Errores posibles:**

| Código | Causa |
|---|---|
| 400 | Formato de imagen no soportado o archivo corrupto |
| 422 | No se proporcionó ningún archivo |
| 500 | Error interno de inferencia |

---

### GET `/metrics`

Métricas en formato Prometheus (scrapeadas automáticamente).

```bash
curl http://localhost:8000/metrics
```

---

### GET `/api`

Información general de la API en JSON.

---

### GET `/docs`

Documentación interactiva Swagger UI.

---

## Métricas Prometheus

Las siguientes métricas se exponen en `/metrics`:

| Métrica | Tipo | Descripción |
|---|---|---|
| `pneumonia_predictions_total` | Counter | Total de predicciones realizadas |
| `pneumonia_predictions_by_class_total` | Counter | Predicciones desglosadas por clase |
| `pneumonia_inference_latency_seconds` | Histogram | Latencia real de inferencia |
| `pneumonia_prediction_errors_total` | Counter | Errores de predicción |
| `pneumonia_active_predictions` | Gauge | Predicciones en curso |

---

## Dashboard Grafana

El dashboard se provisiona automáticamente al levantar el stack. Incluye:

- **Total de predicciones** — contador acumulado
- **Distribución por clase** — porcentaje por diagnóstico
- **Latencia de inferencia** — p50, p95, p99
- **Tasa de errores** — errores por minuto
- **Predicciones por minuto** — throughput en tiempo real

Para visualizar datos, realiza varias predicciones desde el frontend (`http://localhost:8000`) y espera el intervalo de scraping (15 s).

---

## Tests

```bash
# Sin Docker (requiere dependencias instaladas)
pip install -r requirements.txt
python -m pytest tests/ -v

# Con el stack levantado
docker-compose up -d
python -m pytest tests/ -v
```

Estado actual: **13/13 tests pasando**.

Los tests cubren:
- `GET /health` — estado del servicio
- `GET /info` — metadatos del modelo
- `POST /predict` — predicción con imágenes reales de las 3 clases
- Validación de inputs inválidos (formato incorrecto, archivo vacío)
- Wrapper de inferencia y carga del modelo

---

## Alumno

**Rodrigo Gálvez Travalja**  
Grupo: Víctor Teruel, Antonio Gallego, Francisco Javier Garrido, Javier Ramos  
Aplicaciones de Inteligencia Artificial II · Curso 24-25
