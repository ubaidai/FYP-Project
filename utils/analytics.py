from transformers import pipeline
import streamlit as st
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )


def analyze_sentiment(text: str, model) -> Dict:
    try:
        result = model(text[:512])[0]
        label = result['label']
        score = result['score']
        if label == "POSITIVE" and score > 0.7:
            return {"sentiment": "Satisfied", "emoji": "😊", "color": "#00C853", "score": round(score, 2)}
        elif label == "NEGATIVE" and score > 0.7:
            return {"sentiment": "Frustrated", "emoji": "😞", "color": "#FF3D00", "score": round(score, 2)}
        else:
            return {"sentiment": "Neutral", "emoji": "😐", "color": "#FFB300", "score": round(score, 2)}
    except Exception:
        return {"sentiment": "Neutral", "emoji": "😐", "color": "#FFB300", "score": 0.5}


def get_sentiment_summary(conversations: List[Dict]) -> Dict:
    counts = {"Satisfied": 0, "Neutral": 0, "Frustrated": 0}
    for conv in conversations:
        s = conv.get("sentiment", "Neutral")
        if s in counts:
            counts[s] += 1
    return counts


def cluster_unanswered_questions(questions: List[str], n_clusters: int = 3) -> List[Dict]:
    if len(questions) < 3:
        return [{"cluster": 0, "question": q, "topic": "General"} for q in questions]
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(questions)
        n_clusters = min(n_clusters, len(questions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        feature_names = vectorizer.get_feature_names_out()
        cluster_topics = {}
        for cid in range(n_clusters):
            center = kmeans.cluster_centers_[cid]
            top_indices = center.argsort()[-3:][::-1]
            topic = " / ".join([feature_names[i] for i in top_indices])
            cluster_topics[cid] = topic.title()
        return [
            {"cluster": int(labels[i]), "question": questions[i], "topic": cluster_topics[int(labels[i])]}
            for i in range(len(questions))
        ]
    except Exception:
        return [{"cluster": 0, "question": q, "topic": "General"} for q in questions]
