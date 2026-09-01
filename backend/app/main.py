"""FastAPI application entry point for Prism AI Clinical Intelligence."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.patient_route import router as patient_router

app = FastAPI(
    title="Prism AI Clinical Intelligence",
    description="AI-assisted clinical risk detection and care-navigation prototype.",
    version="0.1.0",
)

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8888",
    "http://127.0.0.1:8888",
    "http://localhost:8889",
    "http://127.0.0.1:8889",
]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patient_router)


@app.get("/")
def root():
    return {
        "message": "Prism AI Clinical Intelligence API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "prism-ai-clinical-intelligence"}

