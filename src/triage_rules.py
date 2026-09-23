"""
src/triage_rules.py
Business rules, SLA calculation, and Human-in-the-Loop (HITL) safety gating.
Decoupled from ML inference to allow instant policy adjustments.
"""

from typing import Dict, Any

# SLA mapping based on priority
SLA_HOURS = {
    "Critical": 1,
    "High": 4,
    "Medium": 24,
    "Low": 48
}

# Squad routing mapping
DEPARTMENT_ROUTING = {
    "Payments": "Billing & Payments Squad (Tier 2)",
    "Login": "Identity & Access Management (Auth)",
    "API": "Developer Platform & SDK Team",
    "Mobile": "Mobile Engineering Squad",
    "Dashboard": "Core Web App & Analytics",
    "Other": "General Operations Tier 1"
}

# High-risk sensitive keywords that bypass auto-resolution
HIGH_RISK_KEYWORDS = [
    "refund", "unauthorized", "chargeback", "delete", 
    "sue", "lawyer", "legal", "gdpr", "breach", "hacked"
]


def apply_triage_rules(ticket_text: str, predictions: Dict[str, str], confidences: Dict[str, float]) -> Dict[str, Any]:
    """
    Applies production business logic on top of ML predictions.
    Determines SLA, squad assignment, and Human Review Queue gating.
    """
    priority = predictions.get("priority", "Medium")
    product_area = predictions.get("product_area", "Other")
    sentiment = predictions.get("sentiment", "Neutral")
    
    # 1. Calculate SLA
    sla_hours = SLA_HOURS.get(priority, 24)
    sla_label = f"{sla_hours} hour{'s' if sla_hours > 1 else ''}"

    # 2. Determine Department Routing
    assigned_team = DEPARTMENT_ROUTING.get(product_area, "General Operations Tier 1")

    # 3. Human-in-the-Loop (HITL) Safety Check (Ankush's Review Queue)
    requires_human_review = False
    review_reasons = []

    # Check 1: High Risk keywords
    text_lower = ticket_text.lower()
    for kw in HIGH_RISK_KEYWORDS:
        if kw in text_lower:
            requires_human_review = True
            review_reasons.append(f"High-risk safety keyword detected: '{kw}'")
            break

    # Check 2: Low Confidence on critical decisions (< 0.45)
    for task in ["priority", "product_area"]:
        score = confidences.get(task, 1.0)
        if score < 0.45:
            requires_human_review = True
            review_reasons.append(f"Low confidence on {task} ({score*100:.1f}%)")

    # Check 3: Critical Priority + Negative Sentiment
    if priority == "Critical" and sentiment == "Negative":
        requires_human_review = True
        review_reasons.append("High churn risk: Negative sentiment with Critical priority")

    # 4. Generate Recommended Next Action
    if requires_human_review:
        action = f"Escalate to {assigned_team} lead for manual review."
    else:
        action = f"Auto-route ticket to {assigned_team} queue with {sla_label} SLA."

    return {
        "sla_hours": sla_hours,
        "sla_label": sla_label,
        "assigned_team": assigned_team,
        "requires_human_review": requires_human_review,
        "review_reasons": review_reasons,
        "suggested_action": action
    }
