"""
src/app.py
Production FastAPI Service for Support Ticket Triage.
Exposes REST endpoints, metrics, sample feeds, and serves the UI.
"""
import sys
from pathlib import Path

# Add src/ to sys.path so internal imports resolve cleanly everywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import json
import os

from inference import TriageInferenceEngine

# ====================== APP SETUP ======================
app = FastAPI(
    title="Support Ticket Triage Engine",
    description="Multi-Task DistilBERT Support Triage API (Sentiment, Category, Area, Priority, Satisfaction)",
    version="1.0.0"
)

# Enable CORS for local testing & integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = TriageInferenceEngine()


# ====================== SCHEMAS ======================
class TicketRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Support ticket customer message")

class BatchTicketRequest(BaseModel):
    tickets: List[str] = Field(..., min_items=1, description="List of support tickets to triage")


# ====================== ENDPOINTS ======================

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Liveness & readiness probe for Docker / Kubernetes orchestrators."""
    return {
        "status": "healthy",
        "service": "support-ticket-triage",
        "device": str(engine.model.encoder.device),
        "model": "MultiTaskDistilBERT"
    }


@app.post("/api/v1/triage")
def triage_ticket_endpoint(request: TicketRequest):
    """Triages a single incoming support ticket."""
    try:
        result = engine.triage_ticket(request.text)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/triage/batch")
def triage_batch_endpoint(request: BatchTicketRequest):
    """Triages a batch of incoming tickets for bulk queue processing."""
    try:
        results = engine.triage_batch(request.tickets)
        return {"success": True, "count": len(results), "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
def get_model_metrics():
    """Returns evaluation metrics (Exact Match, Accuracy, F1) for Screen 3."""
    metrics_path = Path("models/multitask_distilbert/test_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return {"success": True, "metrics": json.load(f)}
    return {"success": False, "message": "Metrics not found"}


@app.get("/api/v1/sample-tickets")
def get_sample_tickets():
    """Returns curated tickets from the dataset to populate Ankush's Screen 1 Queue."""
    sample_file = Path("data/processed/test.json")
    if not sample_file.exists():
        sample_file = Path("data/raw/support_tickets.json")
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Return first 15 diverse tickets
            return {"success": True, "tickets": data[:15]}
    
    # Fallback default samples if raw data isn't loaded
    fallback_samples = [
        {"ticket_id": "01", "text": "I can't log in to my account. It keeps saying invalid credentials even after resetting.", "priority": "High"},
        {"ticket_id": "02", "text": "Refund + delete all our candidate data immediately. Unauthorized charge of $500.", "priority": "Critical"},
        {"ticket_id": "03", "text": "The mobile app crashes every time I click the upload attachment button.", "priority": "High"},
        {"ticket_id": "04", "text": "Please add dark mode to the analytics dashboard.", "priority": "Low"},
        {"ticket_id": "05", "text": "API rate limit 429 errors are coming too frequently on our production webhooks.", "priority": "Critical"},
    ]
    return {"success": True, "tickets": fallback_samples}


# Mount Static UI directory
static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
