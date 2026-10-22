import os
from dotenv import load_dotenv

load_dotenv()

# The four specialist queues (= four specialist agents) + human escalation.
QUEUES = {
    "billing": "Invoices, payments, refunds, cancellation fees",
    "orders": "Placing/changing/cancelling orders, delivery, shipping address",
    "account": "Login, profile, registration, subscriptions, newsletter",
    "general": "Contacting the company, feedback, complaints, anything else",
}

# Bitext dataset has 11 categories -> we fold them into 4 queues.
CATEGORY_TO_QUEUE = {
    "INVOICE": "billing", "PAYMENT": "billing", "REFUND": "billing",
    "CANCEL": "billing", "CANCELLATION_FEE": "billing",
    "ORDER": "orders", "DELIVERY": "orders",
    "SHIPPING": "orders", "SHIPPING_ADDRESS": "orders",
    "ACCOUNT": "account", "SUBSCRIPTION": "account",
    "CONTACT": "general", "FEEDBACK": "general",
}

URGENCY_LEVELS = [
    "Routine question, no time pressure",
    "Time-sensitive or customer is clearly bothered",
    "Critical: money lost, service blocked, or very angry customer",
]

# Decision thresholds (tune these on a validation split - good report material).
ROUTE_CONF_MIN = 0.60    # below this the router is "unsure" -> human
HUMAN_NOUL_MIN = 0.80    # P(needs human) above this -> human
GUARD_NOUL_MIN = 0.70    # P(draft is policy compliant) below this -> human


def env(name, default=None):
    return os.getenv(name, default)
