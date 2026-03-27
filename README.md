# 🧠 NeuroScan AI — Explainable Brain Tumor Detection System

> **Explainable and Confidence-Aware Deep Learning System for Brain Tumor Detection and Classification using ResNet101 and Grad-CAM++**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

NeuroScan AI is a full-stack, production-ready clinical decision-support system that:

- **Detects** whether an MRI scan contains a brain tumour (binary classification)
- **Classifies** the tumour type: Glioma, Meningioma, or Pituitary Adenoma
- **Explains** predictions visually using **Grad-CAM++** heatmaps overlaid on the MRI
- **Quantifies** prediction confidence and flags uncertain results
- **Provides** clinical risk insights and follow-up recommendations
- **Secures** all endpoints with JWT authentication and bcrypt password hashing

---

## System Architecture

```
Browser (React + Vite :3000)
        |
        |  REST / multipart
        v
FastAPI Backend (:8000)
   |-- /auth/register   POST
   |-- /auth/login      POST
   |-- /predict         POST  (JWT protected)
             |
             |-- Preprocessing  (PIL -> 224x224 tensor)
             |-- Detection Model  (ResNet101 binary)
             |-- Classification Model  (ResNet101 3-class)
             |-- Grad-CAM++ Engine
             +-- Risk Analysis Service
```

---

## AI Pipeline

```
MRI Upload
    |
    v
Image Preprocessing
  - Resize to 224x224
  - Normalize (ImageNet mean/std)
  - Augmentation (training only)
    |
    v
Tumor Detection Model (ResNet101)
  - BCEWithLogitsLoss, pos_weight=2 (High Recall)
  - Output: P(tumor)
    |
   / \
  /   \
No    Tumor
Tumor   |
  |     v
  |   Tumor Classification Model (ResNet101)
  |   - 3 classes: glioma / meningioma / pituitary
  |   - Softmax confidence score
  |     |
  |     v
  |   Confidence-Aware Decision
  |   - conf >= 0.75: Reliable
  |   - conf <  0.75: Uncertain
  |     |
  |     v
  |   Grad-CAM++ Heatmap
  |   - Hooks into backbone.layer4
  |   - Overlaid on original MRI as base64 PNG
  |     |
  |     v
  +-> Risk Analysis
      - Glioma     -> High Risk
      - Meningioma -> Moderate Risk
      - Pituitary  -> Moderate (Hormonal)
      - None       -> Low Risk
          |
          v
      JSON Response
```

---

## Project Structure

```
brain_tumor_project/
|
|-- README.md
|
|-- backend/
|   |-- main.py                   # FastAPI app + CORS + lifespan
|   |-- config.py                 # Pydantic settings from .env
|   |-- database.py               # Async SQLAlchemy engine
|   |-- train.py                  # Full model training script
|   |-- setup_dataset.py          # Dataset organisation helper
|   |-- requirements.txt
|   |-- .env
|   |
|   |-- auth/
|   |   |-- __init__.py
|   |   |-- models.py             # User ORM model (SQLAlchemy)
|   |   |-- schemas.py            # Pydantic request/response schemas
|   |   |-- service.py            # Hashing, JWT, user CRUD
|   |   |-- dependencies.py       # get_current_user dependency
|   |   +-- router.py             # /auth/register + /auth/login
|   |
|   |-- models/
|   |   +-- resnet_models.py      # TumorDetectionModel + TumorClassificationModel
|   |
|   |-- routes/
|   |   +-- predict.py            # POST /predict endpoint
|   |
|   +-- services/
|       |-- preprocessing.py      # PIL -> tensor pipeline
|       |-- gradcam.py            # Grad-CAM++ implementation
|       |-- model_loader.py       # Singleton model loader
|       |-- predictor.py          # Full pipeline orchestration
|       +-- risk_analysis.py      # Clinical risk mapping
|
+-- frontend/
    |-- index.html
    |-- vite.config.js
    |-- tailwind.config.js
    |-- postcss.config.js
    |-- package.json
    |
    +-- src/
        |-- main.jsx
        |-- App.jsx               # Router setup
        |-- index.css             # Tailwind + custom styles
        |
        |-- components/
        |   |-- AuthContext.jsx   # React auth context
        |   |-- ProtectedRoute.jsx
        |   |-- Navbar.jsx
        |   |-- ConfidenceBar.jsx
        |   |-- RiskBadge.jsx
        |   +-- ResultsPanel.jsx
        |
        |-- pages/
        |   |-- LoginPage.jsx
        |   |-- RegisterPage.jsx
        |   +-- DashboardPage.jsx
        |
        +-- services/
            +-- api.js            # Axios instance + interceptors
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- GPU optional (CPU works fine for inference)

### 1. Clone

```bash
git clone https://github.com/yourusername/neuroscan-ai.git
cd neuroscan-ai
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Edit .env — set a strong SECRET_KEY (32+ characters)

uvicorn main:app --reload --port 8000
```

- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

- App: http://localhost:3000

---

## Dataset Setup

Download from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Then run:
```bash
cd backend
python setup_dataset.py --src /path/to/raw_dataset --dst dataset
```

---

## Model Training

```bash
# Train detection model (tumor / no_tumor)
python train.py --task detection --epochs 30

# Train classification model (glioma / meningioma / pituitary)
python train.py --task classification --epochs 30
```

Saved weights:
- `backend/models/detection_resnet101.pth`
- `backend/models/classification_resnet101.pth`

The server auto-loads them on startup. If weights are missing, it runs in demo mode with ImageNet pretrained backbone.

---

## API Reference

### POST /auth/register

```json
{
  "full_name": "Dr. Priya Sharma",
  "email": "priya@hospital.com",
  "password": "securepass123",
  "role": "user"
}
```

### POST /auth/login

```json
{ "email": "priya@hospital.com", "password": "securepass123" }
```

Response: `{ "access_token": "eyJ...", "role": "user", ... }`

### POST /predict (Protected)

Headers: `Authorization: Bearer <token>`
Body: `multipart/form-data` with `file` field

Response:
```json
{
  "tumor_detected": true,
  "tumor_type": "glioma",
  "confidence": 0.921,
  "reliability": "Reliable",
  "risk_level": "High",
  "risk_color": "red",
  "clinical_note": "...",
  "recommendation": "...",
  "heatmap_image": "data:image/png;base64,..."
}
```

---

## Evaluation Metrics

### Detection (Binary)
| Metric | Target |
|---|---|
| Accuracy | > 95% |
| Recall | > 97% (priority — minimise false negatives) |
| Precision | > 93% |
| F1-Score | > 95% |

### Classification (3-class)
| Metric | Target |
|---|---|
| Accuracy | > 92% |
| Macro F1 | > 90% |

---

## Technologies

**Backend:** FastAPI, PyTorch, TorchVision, SQLAlchemy (async), python-jose, passlib/bcrypt, Pillow, OpenCV

**Frontend:** React 18, Vite, Tailwind CSS, Axios, React Dropzone, Lucide Icons

**AI:** ResNet101, Grad-CAM++, ImageNet pretrained weights, CosineAnnealingLR, AdamW

---

## License

MIT License

---

> Medical Disclaimer: This system is a research and educational tool. It must NOT be used as the sole basis for any clinical decision. Always consult a qualified medical professional.
