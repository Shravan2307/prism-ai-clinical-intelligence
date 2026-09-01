"""FastAPI application entry point for Prism AI Clinical Intelligence."""

from fastapi import FastAPI

from routers.patient_route import router as patient_router

app = FastAPI(
    title="Prism AI Clinical Intelligence",
    description="AI-assisted clinical risk detection and care-navigation prototype.",
    version="0.1.0",
)
app.include_router(patient_router)
