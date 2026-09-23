# Sentry Triage – Production AI Support Ticket Triage Engine 🛡️

An end-to-end, multi-task NLP support triage microservice built with **PyTorch**, **DistilBERT**, **FastAPI**, and **Docker**. 

Features real-time ticket categorization, dynamic SLA tracking, a First-Principles RAG knowledge-base retrieval engine, and a **Human-in-the-Loop (HITL)** review queue to safeguard high-risk decisions.

---

## ⚡ Architectural Highlights

1. **Multi-Task Learning (MTL) Backbone:** One shared `distilbert-base-uncased` transformer trunk with 5 linear classification heads (Sentiment, Category, Product Area, Priority, Satisfaction) running in a single **< 20ms forward pass** on CPU.
2. **First-Principles RAG Engine (Terminal 2):** Deterministic TF-IDF vector space cosine similarity over support documentation, generating grounded response drafts in **< 2ms offline**.
3. **Human-in-the-Loop (HITL) Safety Gate (Screen 2):** Flags uncertain decisions (confidence < 0.45) and sensitive keywords (refunds, GDPR, account deletions) for human review with dynamic SLA countdowns (1h to 48h).
4. **Observability Suite (Screen 3):** Real-time monitoring of multi-head **Exact-Match** accuracy, confusion matrices, and test disagreement logs.
5. **Production Dockerization:** Fully containerized with CPU-optimized PyTorch wheels and `/health` orchestrator probes.

---

## 🚀 Quickstart with Docker

```bash
# 1. Clone repo
git clone https://github.com/disneyatomik/sentry-triage-ticket.git
cd sentry-triage-ticket

# 2. Run with Docker Compose
docker compose up --build -d

# 3. Open Web Dashboard
http://localhost:8000
