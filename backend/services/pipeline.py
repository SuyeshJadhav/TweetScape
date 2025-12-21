"""
Pipeline module - Main orchestration for tweet processing.
Calls each service step in order with clear error handling.
"""
import os
import re
import json
import pandas as pd
import numpy as np
from config import DATA_DIR, setup_logging
from services.analysis import (
    get_embeddings, get_sentiment, get_emotion, 
    extract_topics, get_cluster_keywords
)
from services.clustering_new import reduce_dimensions, assign_clusters_by_sentiment

logger = setup_logging("pipeline")


def deduplicate_tweets(df: pd.DataFrame) -> tuple:
    """
    Remove duplicate and near-duplicate tweets.
    
    Returns:
        (cleaned_df, stats_dict)
    """
    original_count = len(df)
    logger.info(f"Starting deduplication with {original_count} tweets")
    
    # 1. Exact duplicates
    df = df.drop_duplicates(subset=['text'], keep='first')
    exact_dupes = original_count - len(df)
    
    # 2. Near-duplicates (normalize and compare)
    df = df.copy()
    df['text_normalized'] = df['text'].apply(
        lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x.lower().strip())
    )
    df['text_normalized'] = df['text_normalized'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip()
    )
    
    before_near = len(df)
    df = df.drop_duplicates(subset=['text_normalized'], keep='first')
    near_dupes = before_near - len(df)
    
    df = df.drop(columns=['text_normalized'])
    
    stats = {
        'original': original_count,
        'exact_duplicates': exact_dupes,
        'near_duplicates': near_dupes,
        'unique': len(df)
    }
    
    logger.info(f"Deduplication complete: {stats['unique']}/{original_count} unique ({exact_dupes} exact, {near_dupes} near duplicates)")
    
    return df, stats


def process_topic_data(topic: str) -> dict:
    """
    Main pipeline: Load -> Dedupe -> Embed -> Reduce -> Analyze -> Cluster -> Save
    
    Args:
        topic: Search topic string
    
    Returns:
        Result dict with all data and stats
    """
    safe_topic = topic.replace(" ", "_")
    input_file = os.path.join(DATA_DIR, f"tweets_{safe_topic}.json")
    output_file = os.path.join(DATA_DIR, f"clustered_{safe_topic}.json")
    
    # ========== STEP 1: LOAD DATA ==========
    logger.info(f"Loading tweets from {input_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return None
    
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    if len(raw_data) < 3:
        logger.error("Not enough tweets for clustering (need at least 3)")
        return None
    
    df = pd.DataFrame(raw_data)
    logger.info(f"Loaded {len(df)} tweets")
    
    # ========== STEP 2: DEDUPLICATE ==========
    df, dedup_stats = deduplicate_tweets(df)
    
    if len(df) < 3:
        logger.error("Not enough tweets after deduplication")
        return None
    
    # ========== STEP 3: GENERATE EMBEDDINGS ==========
    logger.info("Generating embeddings...")
    embeddings = get_embeddings(df["text"].tolist())
    
    # ========== STEP 4: DIMENSIONALITY REDUCTION ==========
    logger.info("Reducing dimensions with UMAP...")
    coords = reduce_dimensions(embeddings)
    df['x'] = coords[:, 0].tolist()
    df['y'] = coords[:, 1].tolist()
    
    # ========== STEP 5: SENTIMENT ANALYSIS ==========
    logger.info("Analyzing sentiment...")
    sentiments = [get_sentiment(text) for text in df['text'].tolist()]
    df['sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    
    # ========== STEP 6: EMOTION ANALYSIS ==========
    logger.info("Analyzing emotion...")
    emotions = [get_emotion(text) for text in df['text'].tolist()]
    df['emotion'] = [e['emotion'] for e in emotions]
    df['emotion_score'] = [e['score'] for e in emotions]
    
    # ========== STEP 7: CLUSTER ASSIGNMENT ==========
    logger.info("Assigning clusters...")
    df['cluster'] = assign_clusters_by_sentiment(sentiments)
    
    # ========== STEP 8: KEYWORD EXTRACTION ==========
    logger.info("Extracting keywords...")
    cluster_keywords = get_cluster_keywords(df)
    
    # ========== STEP 9: TOPIC EXTRACTION ==========
    logger.info("Extracting topics...")
    topic_results = extract_topics(df['text'].tolist())
    df['topics'] = topic_results['tweet_topics']
    df['primary_topic'] = [t[0] if t else 'unknown' for t in topic_results['tweet_topics']]
    
    # ========== STEP 10: SAVE RESULTS ==========
    output_data = df.to_dict(orient="records")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Pipeline complete! Saved to {output_file}")
    
    return {
        "topic": topic,
        "total_tweets": len(df),
        "clusters": 3,
        "cluster_keywords": cluster_keywords,
        "dedup_stats": dedup_stats,
        "topic_stats": {"top_topics": topic_results['top_topics']},
        "data": output_data
    }
