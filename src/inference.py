"""
src/inference.py
Production Multi-Task Inference Engine.
Runs single and batch forward passes with Softmax confidence scoring
and connects the RAG Grounding Engine (Terminal 2).
"""

import torch
import joblib
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoTokenizer

from model import MultiTaskDistilBERT
from triage_rules import apply_triage_rules
from rag import SupportRAGEngine

# ====================== CONFIG ======================
MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = Path("models/multitask_distilbert/best_model.pt")
ENCODER_DIR = Path("models/label_encoders")
MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_FIELDS = ["sentiment", "request_type", "product_area", "priority", "satisfaction"]


class TriageInferenceEngine:
    def __init__(self):
        print(f"[InferenceEngine] Initializing on {DEVICE}...")
        
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 2. Load Label Encoders
        self.encoders = {}
        for field in LABEL_FIELDS:
            enc_file = ENCODER_DIR / f"{field}_encoder.joblib"
            if not enc_file.exists():
                raise FileNotFoundError(f"Encoder not found: {enc_file}")
            self.encoders[field] = joblib.load(enc_file)

        # 3. Calculate number of classes per task
        num_labels = {field: len(le.classes_) for field, le in self.encoders.items()}

        # 4. Load Model Architecture & Checkpoint
        self.model = MultiTaskDistilBERT(model_name=MODEL_NAME, num_labels=num_labels)
        if MODEL_PATH.exists():
            print(f"[InferenceEngine] Loading fine-tuned weights from {MODEL_PATH}")
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        else:
            print(f"[InferenceEngine] WARNING: Checkpoint not found at {MODEL_PATH}. Using base model.")

        self.model.to(DEVICE)
        self.model.eval()

        # 5. Initialize RAG Knowledge Base Retrieval
        self.rag = SupportRAGEngine()
        print("[InferenceEngine] Ready for low-latency triage + RAG grounding!")

    @torch.inference_mode()
    def triage_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """Triages a single ticket, applies business safety rules, and generates grounded RAG draft."""
        # 1. Tokenize
        encoded = self.tokenizer(
            ticket_text,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # 2. Single forward pass
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # 3. Extract predictions & probabilities
        predictions = {}
        confidences = {}

        for field in LABEL_FIELDS:
            field_logits = logits[field]
            probs = torch.softmax(field_logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            conf = probs[pred_idx].item()

            pred_label = self.encoders[field].inverse_transform([pred_idx])[0]
            predictions[field] = str(pred_label)
            confidences[field] = round(conf, 4)

        # 4. Apply Business & Safety Rules
        triage_meta = apply_triage_rules(ticket_text, predictions, confidences)

        # 5. Retrieve Grounded Support Policy & Draft Response (RAG)
        rag_data = self.rag.retrieve_grounded_response(ticket_text)

        return {
            "ticket_text": ticket_text,
            "predictions": predictions,
            "confidences": confidences,
            "triage": triage_meta,
            "rag": rag_data
        }

    def triage_batch(self, tickets: List[str]) -> List[Dict[str, Any]]:
        """Processes multiple tickets sequentially."""
        return [self.triage_ticket(t) for t in tickets]


# Quick test when running directly
if __name__ == "__main__":
    engine = TriageInferenceEngine()

    sample = "I was charged twice for my subscription this month. Please refund the extra amount."
    result = engine.triage_ticket(sample)
    
    print("\n" + "="*60)
    print(f"Ticket:      \"{result['ticket_text']}\"")
    print(f"Predictions: {result['predictions']}")
    print(f"Confidences: {result['confidences']}")
    print(f"Team:        {result['triage']['assigned_team']}")
    print(f"SLA:         {result['triage']['sla_label']}")
    print(f"RAG Policy:  {result['rag']['doc_title']} (score: {result['rag']['similarity_score']})")
    print(f"RAG Draft:   {result['rag']['draft_response']}")
    print("="*60)
