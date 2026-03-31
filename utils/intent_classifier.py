import re


# Keyword-based intent classification
# In a full project, replace this with a fine-tuned BERT classifier

INTENT_PATTERNS = {
    "billing": [
        r"\bpay(ment|ing|ed)?\b", r"\binvoice\b", r"\bcharge\b", r"\brefund\b",
        r"\bsubscription\b", r"\bprice\b", r"\bcost\b", r"\bplan\b",
        r"\bbill(ing)?\b", r"\bcredit card\b", r"\btransaction\b", r"\bfee\b"
    ],
    "technical": [
        r"\berror\b", r"\bbug\b", r"\bcrash(ing|ed)?\b", r"\bnot work(ing)?\b",
        r"\bissue\b", r"\bproblem\b", r"\bfix\b", r"\binstall(ation)?\b",
        r"\bsetup\b", r"\bconfigur\b", r"\bintegrat\b", r"\bapi\b",
        r"\bpassword\b", r"\blogin\b", r"\baccess\b", r"\bcan't\b", r"\bcannot\b"
    ],
    "returns": [
        r"\breturn\b", r"\bexchange\b", r"\bcancel(lation)?\b", r"\brefund\b",
        r"\bship(ping|ment|ped)?\b", r"\bdeliver(y|ed)?\b", r"\btrack(ing)?\b",
        r"\border\b", r"\bpackage\b", r"\bproduct\b", r"\bitem\b"
    ],
    "general": [
        r"\bhow\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b", r"\bwho\b",
        r"\binfo\b", r"\binformation\b", r"\btell me\b", r"\bexplain\b",
        r"\bhelp\b", r"\bsupport\b", r"\bquestion\b"
    ]
}


def classify_intent(text: str) -> str:
    """
    Classify customer question into one of: billing, technical, returns, general, other.
    Uses regex pattern matching — can be upgraded to a fine-tuned BERT classifier.
    """
    text_lower = text.lower()
    scores = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    # If no patterns matched, return 'other'
    if scores[best_intent] == 0:
        return "other"

    return best_intent
