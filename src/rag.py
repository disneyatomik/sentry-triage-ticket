"""
src/rag.py
Lightweight First-Principles RAG Engine (Terminal 2 in Ankush's design).
Retrieves official support policy excerpts via TF-IDF Vector Space Cosine Similarity
and generates grounded agent response drafts offline in < 2ms.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any

# Official Support Policy Documents (The Knowledge Base)
KNOWLEDGE_BASE = [
    {
        "doc_id": "AUTH-01",
        "title": "Account & Login Troubleshooting",
        "category": "Login",
        "content": "To reset your password, visit https://app.company.com/forgot-password and enter your registered email. If two-factor authentication (2FA) codes are not arriving, check your spam filter or use your emergency recovery keys generated during account setup. Sessions automatically expire after 14 days of inactivity.",
        "template": "Hello! For login or password issues: please visit our password reset portal at /forgot-password. If 2FA codes are delayed, please use your backup recovery codes."
    },
    {
        "doc_id": "BILL-02",
        "title": "Billing & Refund Policy",
        "category": "Payments",
        "content": "Refund requests are processed within 5-7 business days for duplicate charges or failed checkout transactions. If an unauthorized charge occurs, submit your transaction ID immediately. Subscriptions can be cancelled at any time from Settings > Invoices.",
        "template": "Hi there! I understand you are inquiring about a payment or refund. Our billing policy allows automatic refunds within 5-7 business days for failed transactions. We are prioritizing your ticket with Billing Tier 2."
    },
    {
        "doc_id": "API-03",
        "title": "Developer API & Rate Limits",
        "category": "API",
        "content": "The standard API rate limit is 100 requests per minute per API key. When encountering HTTP 429 Too Many Requests, implement exponential backoff with jitter. Ensure Bearer tokens are refreshed every 60 minutes via the /oauth/token endpoint.",
        "template": "Hello! Regarding your API query: HTTP 429 indicates rate limiting (100 req/min limit). Please implement exponential backoff. For token authentication, verify your Bearer header."
    },
    {
        "doc_id": "MOB-04",
        "title": "Mobile Application Diagnostics",
        "category": "Mobile",
        "content": "For Android or iOS app crashes: force close the application, clear cache from device settings, and ensure you are on the latest app release. If upload attachments fail, ensure storage and camera permissions are granted in OS app permissions.",
        "template": "Hi! For mobile app issues: please ensure your app is updated to the latest version and verify camera/storage permissions are enabled in your phone settings."
    },
    {
        "doc_id": "DASH-05",
        "title": "Dashboard & Custom Reports",
        "category": "Dashboard",
        "content": "Dashboard widgets can be customized by dragging the grid handles in the top right. To export analytics reports in CSV or Excel format, click the Export button located on the top right of the reports table.",
        "template": "Hello! To customize widgets or export dashboard data, use the Export CSV/Excel button in the top right of your analytics tab."
    }
]


class SupportRAGEngine:
    def __init__(self):
        # Build Vector Space Matrix over Knowledge Base
        self.docs = KNOWLEDGE_BASE
        self.corpus = [f"{d['title']} {d['category']} {d['content']}" for d in self.docs]
        
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print("[RAGEngine] Knowledge base indexed (5 support policy documents ready).")

    def retrieve_grounded_response(self, query: str) -> Dict[str, Any]:
        """
        Retrieves the top-ranked policy document using Cosine Similarity
        and generates a grounded draft response for the support agent.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_idx = int(similarities.argmax())
        top_score = float(similarities[top_idx])
        matched_doc = self.docs[top_idx]

        # Grounding confidence threshold
        is_grounded = top_score > 0.15

        return {
            "is_grounded": is_grounded,
            "similarity_score": round(top_score, 4),
            "doc_id": matched_doc["doc_id"] if is_grounded else "NONE",
            "doc_title": matched_doc["title"] if is_grounded else "No matching policy",
            "policy_excerpt": matched_doc["content"] if is_grounded else "Ticket requires custom triage.",
            "draft_response": matched_doc["template"] if is_grounded else "Thank you for reaching out. A support engineer is reviewing your request."
        }


# Quick sanity test
if __name__ == "__main__":
    rag = SupportRAGEngine()
    test_queries = [
        "How do I reset my password? It keeps failing.",
        "Charged twice on my credit card! Refund please.",
        "API returning 429 rate limit errors."
    ]
    for q in test_queries:
        res = rag.retrieve_grounded_response(q)
        print(f"\nQuery: {q}")
        print(f"Matched Doc: {res['doc_title']} (score: {res['similarity_score']})")
        print(f"Draft:       {res['draft_response']}")
