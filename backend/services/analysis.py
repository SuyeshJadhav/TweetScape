"""
Analysis module - Sentiment, emotion, and topic extraction.
Each function is isolated for easy debugging.
"""
import re
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from config import setup_logging
from services.models import (
    embedding_model, sentiment_tokenizer, sentiment_model,
    emotion_pipeline, keybert_model
)

logger = setup_logging("analysis")


def get_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def get_sentiment(text: str) -> dict:
    """
    Analyze sentiment using RoBERTa.
    Returns: {'label': 'positive/negative/neutral', 'score': float}
    """
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        
        scores = torch.softmax(outputs.logits, dim=1)[0]
        labels = ['negative', 'neutral', 'positive']
        max_idx = torch.argmax(scores).item()
        
        # Convert to polarity score (-1 to +1)
        polarity = float(scores[2] - scores[0])
        
        return {
            'label': labels[max_idx],
            'score': polarity
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {'label': 'neutral', 'score': 0.0}


def get_emotion(text: str) -> dict:
    """
    Analyze emotion using DistilRoBERTa.
    Returns: {'emotion': str, 'score': float}
    """
    try:
        result = emotion_pipeline(text, truncation=True, max_length=512)
        return {
            'emotion': result[0]['label'],
            'score': round(result[0]['score'], 3)
        }
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        return {'emotion': 'neutral', 'score': 0.0}


def extract_topics(texts: list, n_topics: int = 10) -> dict:
    """
    Extract topics using KeyBERT.
    Returns: {'top_topics': [(topic, count), ...], 'tweet_topics': [[topics], ...]}
    """
    logger.info(f"Extracting topics from {len(texts)} texts")
    
    all_topics = []
    tweet_topics = []
    
    for text in texts:
        try:
            keywords = keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english', 
                top_n=2
            )
            topics = [kw[0] for kw in keywords] if keywords else []
            tweet_topics.append(topics)
            all_topics.extend(topics)
        except Exception as e:
            logger.warning(f"Topic extraction failed for text: {e}")
            tweet_topics.append([])
    
    # Count topic frequencies
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:n_topics]
    logger.info(f"Top topics: {[t[0] for t in top_topics[:5]]}")
    
    return {
        'top_topics': top_topics,
        'tweet_topics': tweet_topics
    }


def get_cluster_keywords(df, n_keywords: int = 5) -> dict:
    """Extract TF-IDF keywords for each cluster."""
    logger.info("Extracting cluster keywords via TF-IDF")
    
    cluster_keywords = {}
    
    for cluster_id in df['cluster'].unique():
        cluster_texts = df[df['cluster'] == cluster_id]['text'].tolist()
        
        if len(cluster_texts) < 2:
            cluster_keywords[str(cluster_id)] = ["insufficient data"]
            continue
        
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            mean_tfidf = tfidf.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            cluster_keywords[str(cluster_id)] = keywords
        except Exception as e:
            logger.warning(f"TF-IDF failed for cluster {cluster_id}: {e}")
            cluster_keywords[str(cluster_id)] = ["error"]
    
    return cluster_keywords
