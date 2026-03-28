# 🧠 NeuroScan AI — Explainable Brain Tumor Detection System

> **Open, Production-Ready Clinical Decision-Support System for Brain Tumor Detection and Classification using EfficientNet-B4 + Ensemble ResNet101 with Grad-CAM++, Score-CAM, EigenCAM, and MC Dropout Uncertainty Estimation**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![No Auth](https://img.shields.io/badge/Auth-None%20(Open%20API)-brightgreen.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange.svg)](https://git-lfs.com)

---

## Overview

NeuroScan AI is a full-stack, open-access clinical decision-support system that:

- **Detects** whether an MRI scan contains a brain tumour (binary classification)
- **Classifies** the tumour type: Glioma, Meningioma, or Pituitary Adenoma
- **Explains** predictions visually using **Grad-CAM++**, **Score-CAM**, and **EigenCAM** heatmaps
- **Quantifies** uncertainty using **Monte Carlo Dropout** across 5 stochastic forward passes
- **Provides** WHO-graded clinical risk insights and follow-up recommendations
- **Generates** downloadable PDF clinical reports
- **Computes** stateless analytics from submitted scan results
- **Requires no login** — fully open API, no authentication, no database

---

## What Changed from v1

| Feature | v1 | v2 (Current) |
|---|---|---|
| Authentication | JWT + bcrypt | ❌ Removed — open API |
| Database | SQLAlchemy + SQLite | ❌ Removed — stateless |
| Detection Model | ResNet101 | ✅ EfficientNet-B4 + Ensemble |
| Classification Model | ResNet101 | ✅ EfficientNet-B4 |
| Explainability | Grad-CAM++ only | ✅ Grad-CAM++ + Score-CAM + EigenCAM |
| Uncertainty | None | ✅ MC Dropout (5 passes) |
| Preprocessing | Basic resize + normalize | ✅ CLAHE + Skull Strip + TTA |
| PDF Reports | Auth-gated | ✅ Open endpoint |
| Analytics | DB-backed per-user | ✅ Stateless from request body |
| Training | Single train.py | ✅ Separate train_detection.py + train_classification.py |

---

## System Architecture

```
Browser (React + Vite :3000)
        |
        |  REST / multipart  (no token required)
        v
FastAPI Backend (:8000)
   |-- POST /predict/         — MRI upload → full AI pipeline
   |-- POST /report/          — prediction data → PDF download
   +-- POST /analytics/summary — scan list → statistics
```

---

## AI Pipeline

```
MRI Upload (JPEG / PNG / BMP)
    |
    v
Advanced Preprocessing
  ├── Resize to 224×224 (LANCZOS)
  ├── CLAHE — local contrast enhancement (tumour boundary sharpening)
  ├── Skull Strip Simulation — removes bone/skull ring
  ├── ImageNet Normalization — mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
  └── TTA — 3 versions: original + h-flip + brightness shift
    |
    v
Tumor Detection (EfficientNet-B4 + Ensemble ResNet101)
  ├── MC Dropout — 5 stochastic passes → mean probability + std uncertainty
  ├── HIGH_THR >= 0.4  → tumor_detected = True
  ├── LOW_THR  <= 0.15 → tumor_detected = False
  └── FN Safety — if prob > 0.3 and uncertainty > 0.2 → force True
    |
   / \
  /   \
No    Tumor Detected
Tumor   |
  |     v
  |   Tumor Classification (EfficientNet-B4)
  |   ├── 3 classes: glioma / meningioma / pituitary
  |   └── Softmax confidence score
  |     |
  |     v
  |   Reliability Assessment
  |   ├── status=LOW  → reliability = LOW  (uncertain zone)
  |   ├── cls_conf < 0.6 → reliability = MEDIUM
  |   └── cls_conf >= 0.6 → reliability = HIGH
  |     |
  |     v
  |   Multi-Method Explainability
  |   ├── Grad-CAM++  — gradient-weighted activation maps
  |   ├── Score-CAM   — perturbation-based, gradient-free
  |   └── EigenCAM    — PCA on feature maps
  |     |
  |     v
  +-> WHO-Graded Risk Analysis
      ├── Glioma     → High Risk    (WHO II–IV, urgent)
      ├── Meningioma → Moderate Risk (WHO I–III, semi-urgent)
      ├── Pituitary  → Moderate Risk (WHO I, hormonal)
      └── None       → No Risk      (routine follow-up)
          |
          v
      JSON Response + base64 Heatmap PNG
```

---

## Project Structure

```
brain_tumor_project/
|
|-- README.md
|-- .gitignore
|-- .gitattributes              # Git LFS tracking rules
|-- docker-compose.yml
|
|-- Training/                   # Raw dataset — tracked via Git LFS
|   |-- glioma/
|   |-- meningioma/
|   |-- notumor/
|   └── pituitary/
|
|-- Testing/                    # Raw dataset — tracked via Git LFS
|   |-- glioma/
|   |-- meningioma/
|   |-- notumor/
|   └── pituitary/
|
|-- backend/
|   |-- main.py                 # FastAPI app + CORS + GZip + lifespan
|   |-- config.py               # Pydantic settings (model paths, thresholds)
|   |-- requirements.txt
|   |-- setup_dataset.py        # Raw → train/val/test split builder
|   |-- train_detection.py      # EfficientNet-B4 detection training
|   |-- train_classification.py # EfficientNet-B4 classification training
|   |-- generate_graphs.py      # Training metrics visualisation
|   |-- Dockerfile
|   |-- .env
|   |
|   |-- models/
|   |   |-- advanced_models.py          # EfficientNet-B4 detection + classification
|   |   |-- resnet_models.py            # ResNet101 + Ensemble + aliases
|   |   |-- detection_efficientnet.pth  # Trained detection weights (Git LFS)
|   |   |-- classification_model.pth    # Trained classification weights (Git LFS)
|   |   |-- metrics.json                # Detection training metrics
|   |   |-- classification_metrics.json # Classification training metrics
|   |   └── *.png                       # Training curves + confusion matrices
|   |
|   |-- routes/
|   |   |-- predict.py          # POST /predict/
|   |   |-- report.py           # POST /report/
|   |   └── analytics.py        # POST /analytics/summary
|   |
|   └── services/
|       |-- preprocessing.py    # CLAHE + skull strip + TTA + normalization
|       |-- gradcam.py          # Grad-CAM++ + Score-CAM + EigenCAM
|       |-- model_loader.py     # Thread-safe singleton model loader
|       |-- predictor.py        # Full pipeline orchestration + MC Dropout
|       |-- risk_analysis.py    # WHO-graded risk + confidence upgrading
|       └── report_generator.py # FPDF2 clinical PDF generation
|
└── frontend/
    |-- index.html
    |-- vite.config.js
    |-- tailwind.config.js
    |-- package.json
    |
    └── src/
        |-- main.jsx
        |-- App.jsx
        |-- index.css
        |
        |-- components/
        |   |-- Navbar.jsx
        |   |-- ConfidenceBar.jsx
        |   |-- RiskBadge.jsx
        |   └── ResultsPanel.jsx
        |
        |-- pages/
        |   └── DashboardPage.jsx
        |
        └── services/
            └── api.js          # Axios instance — predict, report, analytics
```

---

## Dataset

| Task | Split | Class | Count |
|---|---|---|---|
| Detection | Train | tumor | 2,475 |
| Detection | Train | no_tumor | 395 |
| Detection | Val | tumor | 115 |
| Detection | Val | no_tumor | 105 |
| Classification | Train | glioma | 826 |
| Classification | Train | meningioma | 822 |
| Classification | Train | pituitary | 827 |
| Classification | Val | glioma | 100 |
| Classification | Val | meningioma | 115 |
| Classification | Val | pituitary | 74 |
| **Total** | | | **5,854 images** |

Source: [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git LFS installed (`https://git-lfs.com`)
- GPU optional (CPU works for inference)

### 1. Clone

```bash
git clone https://github.com/vijay1845gs/brain_tumor_project.git
cd brain_tumor_project
git lfs pull   # downloads .pth weights + dataset images
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
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

## Dataset Setup (if rebuilding from raw)

Raw `Training/` and `Testing/` folders are included via Git LFS.

```bash
cd backend
venv\Scripts\python.exe setup_dataset.py --src "C:\path\to\brain_tumor_project" --dst dataset
```

This creates:
```
dataset/
├── detection/
│   ├── train/  (tumor / no_tumor)
│   └── val/    (tumor / no_tumor)
└── classification/
    ├── train/  (glioma / meningioma / pituitary)
    └── val/    (glioma / meningioma / pituitary)
```

---

## Model Training

```bash
cd backend

# Train detection model (EfficientNet-B4)
venv\Scripts\python.exe train_detection.py --data dataset/detection --model efficientnet --epochs 20 --out models

# Train classification model (EfficientNet-B4)
venv\Scripts\python.exe train_classification.py --data dataset/classification --model efficientnet --epochs 25 --out models
```

Saved weights:
- `backend/models/detection_efficientnet.pth`
- `backend/models/classification_model.pth`

The server auto-loads EfficientNet-B4 weights on startup. Falls back to ResNet101 if EfficientNet weights are missing. Falls back to ImageNet pretrained if no weights found (demo mode).

---

## API Reference

### POST /predict/

Upload an MRI image and receive a full AI analysis.

**Request:** `multipart/form-data` with `file` field (JPEG/PNG/BMP)

**Response:**
```json
{
  "tumor_detected": true,
  "tumor_type": "glioma",
  "confidence": 0.921,
  "uncertainty": 0.043,
  "reliability": "HIGH",
  "all_class_probs": [0.921, 0.054, 0.025]
}
```

---

### POST /report/

Generate a downloadable PDF clinical report from prediction data.

**Request:**
```json
{
  "user_name": "Dr. Smith",
  "tumor_detected": true,
  "tumor_type": "glioma",
  "confidence": 0.921,
  "uncertainty": 0.043,
  "reliability": "HIGH",
  "risk_level": "High",
  "clinical_note": "...",
  "recommendation": "...",
  "heatmap_image": "data:image/png;base64,..."
}
```

**Response:** `application/pdf` — `neuroscan_report.pdf`

---

### POST /analytics/summary

Compute statistics from a list of scan results (stateless — no DB).

**Request:**
```json
{
  "scans": [
    { "tumor_detected": true, "tumor_type": "glioma", "confidence": 0.92, "risk_level": "High" },
    { "tumor_detected": false, "tumor_type": null, "confidence": 0.11, "risk_level": "None" }
  ]
}
```

**Response:**
```json
{
  "total_scans": 2,
  "tumor_detected": 1,
  "no_tumor": 1,
  "detection_rate": 0.5,
  "average_confidence": 0.515,
  "tumor_type_distribution": { "glioma": 1 },
  "risk_distribution": { "High": 1, "None": 1 }
}
```

---

## Model Performance

### Detection (EfficientNet-B4)
| Metric | Result |
|---|---|
| Accuracy | > 95% |
| Recall | > 97% (minimise false negatives) |
| Precision | > 93% |
| F1-Score | > 95% |

### Classification (EfficientNet-B4)
| Metric | Result |
|---|---|
| Accuracy | > 92% |
| Weighted F1 | > 90% |

---

## Technologies

**Backend:** FastAPI, PyTorch 2.2, TorchVision, EfficientNet-B4, ResNet101, FPDF2, Pillow, OpenCV, NumPy, scikit-learn

**Frontend:** React 18, Vite, Tailwind CSS, Axios

**AI:** EfficientNet-B4, ResNet101, Ensemble Detection, Grad-CAM++, Score-CAM, EigenCAM, MC Dropout, CLAHE, Skull Stripping, TTA, FocalLoss, AdamW

**DevOps:** Docker, Git LFS, GitHub

---

## Git LFS

Large files are tracked via Git LFS:

| Pattern | Type |
|---|---|
| `*.pth` | Model weights (~250MB total) |
| `Training/**` | Raw training images (~185MB) |
| `Testing/**` | Raw testing images |

After cloning, run `git lfs pull` to download all large files.

---

## License

MIT License

---

> **Medical Disclaimer:** This system is a research and educational tool. It must NOT be used as the sole basis for any clinical decision. Always consult a qualified medical professional.
