"""
main.py — FastAPI application entry point
─────────────────────────────────────────
Run with:
  uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    print("🧠 Brain Tumor Detection System starting up …")
    print("✅ System ready.")
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("👋 Shutting down …")


app = FastAPI(
    title="Brain Tumor Detection & Classification API",
    description=(
        "Advanced Explainable AI System — ResNet101 + Grad-CAM++ + Bayesian Uncertainty. "
        "Public tumor detection and analysis. No authentication required."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "Brain Tumor Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "features": ["prediction", "history", "analytics", "admin", "pdf-reports"],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
