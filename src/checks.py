"""Deterministic checks — everything scoreable without a judge.

intent/urgency are enums and product is a closed vocabulary, so those three
fields are exact-match scored. Only requested_action (free text) needs the
LLM judge. Design decision: judge nothing a string comparison can score;
judges drift, string equality doesn't.
"""
import json

INTENTS = {"refund_request", "cancellation", "technical_issue", "billing_dispute",
           "feature_request", "account_access", "order_status", "general_inquiry"}
URGENCIES = {"low", "medium", "high"}
PRODUCTS = {"PulseBoard", "Streamline CRM", "Nimbus Backup", "Quanta Analytics",
            "Relay Mail", "Vantage POS", "Orbit Scheduler", "Cascade Forms"}
FIELDS = ["intent", "urgency", "product", "requested_action"]


def parse_output(raw: str):
    """Parse model output to a dict, tolerating code fences. None = format failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def check_case(raw: str, gold: dict) -> dict:
    """Returns format/enum validity plus exact-match scores for closed fields."""
    obj = parse_output(raw)
    if obj is None or not all(f in obj for f in FIELDS):
        return {"format_ok": False, "enum_ok": False,
                "intent_match": 0, "urgency_match": 0, "product_match": 0, "parsed": None}
    enum_ok = (obj.get("intent") in INTENTS and obj.get("urgency") in URGENCIES
               and (obj.get("product") is None or obj.get("product") in PRODUCTS))
    return {
        "format_ok": True,
        "enum_ok": enum_ok,
        "intent_match": int(obj.get("intent") == gold["intent"]),
        "urgency_match": int(obj.get("urgency") == gold["urgency"]),
        "product_match": int(obj.get("product") == gold["product"]),
        "parsed": obj,
    }
