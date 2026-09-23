import json
import random
import uuid
from pathlib import Path
from collections import Counter

# ====================== CONFIG ======================
TOTAL_TICKETS = 7000          # Total tickets to generate
OUTPUT_FILE = "data/raw/support_tickets.json"
SEED = 42

random.seed(SEED)

# ====================== LABEL SPACES ======================
SENTIMENTS = ["Positive", "Neutral", "Negative"]
REQUEST_TYPES = ["Bug", "Feature Request", "Question", "Billing", "Account", "Other"]
PRODUCT_AREAS = ["Login", "Payments", "Dashboard", "API", "Mobile", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
SATISFACTIONS = ["Satisfied", "Dissatisfied"]

# ====================== TEMPLATES ======================
# We use templates + variations so the language feels natural

TEMPLATES = {
    "Bug": {
        "Login": [
            "I can't log in to my account. It keeps saying invalid credentials even after resetting the password.",
            "Login page is broken. The submit button does nothing when I click it.",
            "I'm getting a 500 error every time I try to sign in.",
            "Two-factor authentication is not sending me the code."
        ],
        "Payments": [
            "Payment failed but the amount was still deducted from my account.",
            "I was charged twice for the same subscription.",
            "The payment gateway is throwing an error at checkout.",
            "Refund is not reflecting in my account even after 7 days."
        ],
        "Dashboard": [
            "The dashboard is completely blank after login.",
            "Charts on the analytics page are not loading.",
            "I see incorrect numbers on the overview page.",
            "The page keeps refreshing in a loop."
        ],
        "API": [
            "API is returning 401 even with a valid token.",
            "The webhook is not firing on status change.",
            "Rate limit error is coming too frequently.",
            "Response time of the API has become very slow."
        ],
        "Mobile": [
            "The mobile app crashes as soon as I open it.",
            "I can't upload images from the mobile app.",
            "Push notifications stopped working.",
            "App is stuck on the splash screen."
        ],
        "Other": [
            "Something is broken but I can't figure out exactly what.",
            "The system is behaving strangely since the last update."
        ]
    },
    "Feature Request": {
        "Login": [
            "Please add Google login support.",
            "It would be great if we had passwordless login using magic link.",
            "Can you add biometric login for the mobile app?"
        ],
        "Payments": [
            "Please support UPI payments.",
            "We need the ability to save multiple cards.",
            "Can you add invoice download in PDF format?"
        ],
        "Dashboard": [
            "Please add dark mode to the dashboard.",
            "It would help if we could customize the widgets.",
            "Can you add export to Excel for all reports?"
        ],
        "API": [
            "Please provide a Python SDK.",
            "We need bulk endpoints for creating records.",
            "Can you add filtering by date range in the list API?"
        ],
        "Mobile": [
            "Please add offline mode in the mobile app.",
            "It would be great to have fingerprint unlock.",
            "Can you add multi-language support?"
        ],
        "Other": [
            "Please add a way to schedule reports.",
            "It would be useful to have audit logs."
        ]
    },
    "Question": {
        "Login": [
            "How do I reset my password?",
            "Where can I enable two-factor authentication?",
            "How long does the login session last?"
        ],
        "Payments": [
            "How do I update my billing information?",
            "What payment methods do you support?",
            "How can I download my invoices?"
        ],
        "Dashboard": [
            "How do I interpret the conversion metric?",
            "Can I share the dashboard with my team?",
            "How often is the data refreshed?"
        ],
        "API": [
            "What is the rate limit for the free plan?",
            "How do I authenticate API requests?",
            "Where can I find the API documentation?"
        ],
        "Mobile": [
            "Is the mobile app available on iOS?",
            "How do I enable notifications?",
            "Can I use the same account on multiple devices?"
        ],
        "Other": [
            "How do I contact support?",
            "Where can I find the status page?"
        ]
    },
    "Billing": {
        "Payments": [
            "I was charged incorrectly this month.",
            "Please cancel my subscription and process the refund.",
            "Why did my plan automatically renew?",
            "I need a GST invoice for the last payment."
        ],
        "Other": [
            "I want to change my billing cycle from monthly to yearly.",
            "Can I get a discount for annual payment?"
        ]
    },
    "Account": {
        "Login": [
            "I need to change the email associated with my account.",
            "Please help me recover my account. I no longer have access to the old email."
        ],
        "Other": [
            "I want to delete my account permanently.",
            "How can I transfer ownership of the account?",
            "Please update my company name on the account."
        ]
    },
    "Other": {
        "Other": [
            "I have some general feedback about the product.",
            "The recent update has changed a lot of things. Just wanted to share my thoughts."
        ]
    }
}

# Positive / Neutral / Negative phrasing modifiers
POSITIVE_PHRASES = [
    "Overall I'm happy with the product, but ",
    "Love the platform! Just one small thing – ",
    "Everything is working great except ",
    "Thanks for the great service. One request: "
]

NEGATIVE_PHRASES = [
    "This is really frustrating. ",
    "I've been facing this issue for days. ",
    "Extremely disappointed. ",
    "This is unacceptable. ",
    "I'm about to cancel if this is not fixed. "
]

NEUTRAL_PHRASES = [
    "",
    "Hi team, ",
    "Hello, ",
    "I have a query regarding "
]

def generate_ticket_text(request_type: str, product_area: str, sentiment: str) -> str:
    """Generate natural-sounding ticket text."""
    templates = TEMPLATES.get(request_type, {}).get(product_area)
    if not templates:
        templates = TEMPLATES.get(request_type, {}).get("Other", ["I need help with something."])

    base = random.choice(templates)

    if sentiment == "Positive":
        prefix = random.choice(POSITIVE_PHRASES)
        text = prefix + base[0].lower() + base[1:]
    elif sentiment == "Negative":
        prefix = random.choice(NEGATIVE_PHRASES)
        text = prefix + base
    else:
        prefix = random.choice(NEUTRAL_PHRASES)
        text = prefix + base

    # Add some variation
    if random.random() < 0.3:
        text += " Please look into this as soon as possible."
    if random.random() < 0.2:
        text += " Ticket ID reference: " + str(uuid.uuid4())[:8]

    return text.strip()


def decide_priority(request_type: str, sentiment: str) -> str:
    """Simple but realistic priority logic."""
    if request_type == "Bug" and sentiment == "Negative":
        return random.choices(["High", "Critical"], weights=[0.6, 0.4])[0]
    if request_type == "Billing" and sentiment == "Negative":
        return random.choices(["High", "Medium"], weights=[0.7, 0.3])[0]
    if request_type == "Feature Request":
        return random.choices(["Low", "Medium"], weights=[0.7, 0.3])[0]
    if request_type == "Question":
        return random.choices(["Low", "Medium"], weights=[0.8, 0.2])[0]
    return random.choice(["Low", "Medium", "High"])


def decide_satisfaction(sentiment: str) -> str:
    if sentiment == "Positive":
        return "Satisfied"
    if sentiment == "Negative":
        return "Dissatisfied"
    return random.choice(["Satisfied", "Dissatisfied"])


def generate_one_ticket() -> dict:
    request_type = random.choice(REQUEST_TYPES)
    
    # Make product area somewhat correlated with request type
    if request_type in ["Billing"]:
        product_area = random.choice(["Payments", "Other"])
    elif request_type == "Account":
        product_area = random.choice(["Login", "Other"])
    else:
        product_area = random.choice(PRODUCT_AREAS)

    sentiment = random.choices(
        SENTIMENTS,
        weights=[0.25, 0.35, 0.40]  # Slightly more negative (realistic for support)
    )[0]

    text = generate_ticket_text(request_type, product_area, sentiment)
    priority = decide_priority(request_type, sentiment)
    satisfaction = decide_satisfaction(sentiment)

    return {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "text": text,
        "sentiment": sentiment,
        "request_type": request_type,
        "product_area": product_area,
        "priority": priority,
        "satisfaction": satisfaction
    }


def main():
    print(f"Generating {TOTAL_TICKETS} support tickets...")
    tickets = [generate_one_ticket() for _ in range(TOTAL_TICKETS)]

    # Create output directory
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2, ensure_ascii=False)

    # Print distribution
    print("\n=== Label Distribution ===")
    for field in ["sentiment", "request_type", "product_area", "priority", "satisfaction"]:
        counts = Counter(t[field] for t in tickets)
        print(f"\n{field.upper()}:")
        for k, v in counts.most_common():
            print(f"  {k:20} {v:5} ({v/TOTAL_TICKETS*100:.1f}%)")

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()