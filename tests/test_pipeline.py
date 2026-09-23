"""
tests/test_pipeline.py
Automated CI/CD Unit & Integration Tests for Sentry Triage.
"""
import sys
from pathlib import Path

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from triage_rules import apply_triage_rules
from rag import SupportRAGEngine


def test_sla_calculation():
    """Test that Critical priority maps to 1 hour SLA."""
    predictions = {"priority": "Critical", "product_area": "Payments", "sentiment": "Neutral"}
    confidences = {"priority": 0.85, "product_area": 0.90}
    meta = apply_triage_rules("Test ticket", predictions, confidences)
    
    assert meta["sla_hours"] == 1
    assert meta["assigned_team"] == "Billing & Payments Squad (Tier 2)"


def test_human_escalation_on_high_risk_keyword():
    """Test that sensitive keywords trigger human review."""
    predictions = {"priority": "Low", "product_area": "Login", "sentiment": "Neutral"}
    confidences = {"priority": 0.95, "product_area": 0.95}
    meta = apply_triage_rules("I demand a refund immediately", predictions, confidences)
    
    assert meta["requires_human_review"] is True
    assert any("refund" in r for r in meta["review_reasons"])


def test_rag_knowledge_retrieval():
    """Test that RAG engine retrieves grounded policy excerpts."""
    rag = SupportRAGEngine()
    result = rag.retrieve_grounded_response("How do I reset my account password?")
    
    assert result["is_grounded"] is True
    assert result["doc_id"] == "AUTH-01"
    assert "password" in result["draft_response"].lower()
