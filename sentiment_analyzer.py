from transformers import pipeline
import re
from collections import Counter

# Load once and cache
_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipeline


def analyze_sentiment(text: str) -> dict:
    """Returns {label: POSITIVE/NEGATIVE, score: float, emoji: str}"""
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.5, "emoji": "😐"}

    pipe = _get_pipeline()
    result = pipe(text[:512])[0]
    label = result["label"]
    score = result["score"]

    # Map to 3-class with neutral zone
    if label == "POSITIVE" and score > 0.65:
        return {"label": "POSITIVE", "score": score, "emoji": "😊"}
    elif label == "NEGATIVE" and score > 0.65:
        return {"label": "NEGATIVE", "score": score, "emoji": "😞"}
    else:
        return {"label": "NEUTRAL", "score": score, "emoji": "😐"}


def analyze_conversation(messages: list) -> dict:
    """
    Analyze all user messages in a conversation.
    messages: list of {role, content, sentiment} dicts
    Returns summary analytics dict.
    """
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        return {
            "total_messages": 0,
            "sentiment_counts": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "overall_sentiment": "NEUTRAL",
            "overall_emoji": "😐",
            "avg_score": 0.5,
            "satisfaction_pct": 50,
        }

    sentiments = []
    for msg in user_messages:
        if "sentiment" in msg:
            sentiments.append(msg["sentiment"]["label"])
        else:
            result = analyze_sentiment(msg["content"])
            sentiments.append(result["label"])

    counts = Counter(sentiments)
    sentiment_counts = {
        "POSITIVE": counts.get("POSITIVE", 0),
        "NEUTRAL": counts.get("NEUTRAL", 0),
        "NEGATIVE": counts.get("NEGATIVE", 0),
    }

    total = len(sentiments)
    pos_pct = (sentiment_counts["POSITIVE"] / total) * 100 if total else 50

    if sentiment_counts["POSITIVE"] > sentiment_counts["NEGATIVE"]:
        overall = "POSITIVE"
        emoji = "😊"
    elif sentiment_counts["NEGATIVE"] > sentiment_counts["POSITIVE"]:
        overall = "NEGATIVE"
        emoji = "😞"
    else:
        overall = "NEUTRAL"
        emoji = "😐"

    return {
        "total_messages": total,
        "sentiment_counts": sentiment_counts,
        "overall_sentiment": overall,
        "overall_emoji": emoji,
        "avg_score": pos_pct / 100,
        "satisfaction_pct": round(pos_pct),
    }


def get_unanswered_topics(unanswered_questions: list) -> list:
    """
    Simple keyword clustering for unanswered questions.
    Groups by most common keywords to surface knowledge gaps.
    """
    if not unanswered_questions:
        return []

    stop_words = {"the", "a", "an", "is", "it", "in", "on", "at", "to",
                  "for", "of", "and", "or", "i", "my", "your", "do", "does",
                  "can", "how", "what", "when", "where", "why", "this", "that"}

    keyword_map = {}
    for q in unanswered_questions:
        words = re.findall(r'\b[a-z]{3,}\b', q.lower())
        keywords = [w for w in words if w not in stop_words]
        for kw in keywords:
            if kw not in keyword_map:
                keyword_map[kw] = []
            keyword_map[kw].append(q)

    # Return top recurring topics
    sorted_topics = sorted(keyword_map.items(), key=lambda x: len(x[1]), reverse=True)
    return [
        {"keyword": kw, "count": len(qs), "example": qs[0]}
        for kw, qs in sorted_topics[:8]
        if len(qs) >= 1
    ]
