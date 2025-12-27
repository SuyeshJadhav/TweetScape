"""
TweetScape Performance Benchmark Suite
======================================
Measures and quantifies performance metrics for resume bullet points.

Run: python benchmark.py
"""

import time
import json
import os
import sys
import statistics
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

NUM_WARMUP_RUNS = 1
NUM_BENCHMARK_RUNS = 5
SAMPLE_SIZES = [10, 50, 100, 200, 500]

# Sample tweet texts for benchmarking
SAMPLE_TWEETS = [
    "This new AI feature is absolutely amazing! Can't believe how fast it works 🚀",
    "Terrible update. App keeps crashing and losing my data. Very disappointed.",
    "Just tried the new interface - feels pretty standard, nothing special.",
    "The customer support team was incredibly helpful solving my issue! Thank you!",
    "Why did they remove the dark mode? This is so frustrating 😤",
    "Interesting approach to the problem, but I'm not sure it will scale well.",
    "Best purchase I've ever made! Highly recommend to everyone.",
    "The performance improvements are noticeable. Loading times cut in half!",
    "New feature announcement looks promising. Excited to try it out!",
    "Had high hopes but honestly underwhelmed by the execution.",
    "Love the new design! Clean, modern, and intuitive interface.",
    "This update broke everything. Please fix ASAP!",
    "Not bad, but there's definitely room for improvement.",
    "Revolutionary product that will change the industry forever!",
    "Waste of money. Returning it tomorrow.",
    "The team really listened to feedback. Great improvements!",
    "Meh. It's okay I guess. Nothing to write home about.",
    "Absolutely disgusted by the lack of features. Unacceptable.",
    "This solves a problem I didn't know I had. Brilliant!",
    "Fear of missing out on this deal! Limited time only!"
]


def generate_test_data(n: int) -> list:
    """Generate n synthetic tweets for benchmarking."""
    tweets = []
    for i in range(n):
        tweets.append({
            "handle": f"@user_{i}",
            "text": SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)],
            "timestamp": datetime.now().isoformat()
        })
    return tweets


class BenchmarkTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000  # Convert to ms


def format_results(times: list) -> dict:
    """Calculate statistics for a list of timing measurements."""
    return {
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "avg_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "std_ms": round(statistics.stdev(times), 2) if len(times) > 1 else 0
    }


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_embedding_generation():
    """Benchmark sentence embedding generation with all-MiniLM-L6-v2."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Embedding Generation (all-MiniLM-L6-v2)")
    print("="*60)
    
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = {}
    
    for n in SAMPLE_SIZES:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        times = []
        
        # Warmup
        for _ in range(NUM_WARMUP_RUNS):
            _ = embedding_model.encode(texts[:5], show_progress_bar=False)
        
        # Benchmark
        for _ in range(NUM_BENCHMARK_RUNS):
            with BenchmarkTimer(f"embed_{n}") as timer:
                embeddings = embedding_model.encode(texts, show_progress_bar=False)
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["throughput_per_sec"] = round(n / (stats["avg_ms"] / 1000), 1)
        results[f"{n}_texts"] = stats
        
        print(f"  {n:4} texts: {stats['avg_ms']:7.2f}ms avg | {stats['throughput_per_sec']:6.1f} texts/sec")
    
    return results


def benchmark_sentiment_analysis():
    """Benchmark RoBERTa sentiment analysis."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Sentiment Analysis (twitter-roberta)")
    print("="*60)
    
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def get_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]
        labels = ['negative', 'neutral', 'positive']
        max_idx = torch.argmax(scores).item()
        return {'label': labels[max_idx], 'score': float(scores[2] - scores[0])}
    
    results = {}
    single_times = []
    
    # Single text benchmark
    for _ in range(NUM_WARMUP_RUNS):
        _ = get_sentiment(SAMPLE_TWEETS[0])
    
    for _ in range(NUM_BENCHMARK_RUNS * 10):
        with BenchmarkTimer("single_sentiment") as timer:
            _ = get_sentiment(SAMPLE_TWEETS[0])
        single_times.append(timer.elapsed)
    
    results["single_text"] = format_results(single_times)
    print(f"  Single text: {results['single_text']['avg_ms']:.2f}ms avg")
    
    # Batch benchmark
    for n in [10, 50, 100]:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            with BenchmarkTimer(f"sentiment_{n}") as timer:
                for text in texts:
                    _ = get_sentiment(text)
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["throughput_per_sec"] = round(n / (stats["avg_ms"] / 1000), 1)
        results[f"batch_{n}"] = stats
        
        print(f"  {n:4} texts: {stats['avg_ms']:7.2f}ms avg | {stats['throughput_per_sec']:.1f} texts/sec")
    
    return results


def benchmark_emotion_analysis():
    """Benchmark DistilRoBERTa emotion classification."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Emotion Analysis (distilroberta 7-class)")
    print("="*60)
    
    from transformers import pipeline
    
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    
    def get_emotion(text):
        result = emotion_pipeline(text, truncation=True, max_length=512)
        return {'emotion': result[0]['label'], 'score': round(result[0]['score'], 3)}
    
    results = {}
    single_times = []
    
    # Single text benchmark
    for _ in range(NUM_WARMUP_RUNS):
        _ = get_emotion(SAMPLE_TWEETS[0])
    
    for _ in range(NUM_BENCHMARK_RUNS * 10):
        with BenchmarkTimer("single_emotion") as timer:
            _ = get_emotion(SAMPLE_TWEETS[0])
        single_times.append(timer.elapsed)
    
    results["single_text"] = format_results(single_times)
    print(f"  Single text: {results['single_text']['avg_ms']:.2f}ms avg")
    
    # Batch benchmark
    for n in [10, 50, 100]:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            with BenchmarkTimer(f"emotion_{n}") as timer:
                for text in texts:
                    _ = get_emotion(text)
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["throughput_per_sec"] = round(n / (stats["avg_ms"] / 1000), 1)
        results[f"batch_{n}"] = stats
        
        print(f"  {n:4} texts: {stats['avg_ms']:7.2f}ms avg | {stats['throughput_per_sec']:.1f} texts/sec")
    
    return results


def benchmark_umap_reduction():
    """Benchmark UMAP dimensionality reduction."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: UMAP Dimensionality Reduction (384D → 2D)")
    print("="*60)
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = {}
    
    for n in [50, 100, 200, 500]:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        
        # Pre-generate embeddings
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, n-1))
            
            with BenchmarkTimer(f"umap_{n}") as timer:
                coords = reducer.fit_transform(embeddings)
            times.append(timer.elapsed)
        
        stats = format_results(times)
        results[f"{n}_points"] = stats
        
        print(f"  {n:4} points: {stats['avg_ms']:7.2f}ms avg")
    
    return results


def benchmark_keybert_extraction():
    """Benchmark KeyBERT topic/keyword extraction."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: KeyBERT Topic Extraction")
    print("="*60)
    
    from keybert import KeyBERT
    
    kw_model = KeyBERT()
    results = {}
    
    # Single text
    single_times = []
    for _ in range(NUM_WARMUP_RUNS):
        _ = kw_model.extract_keywords(SAMPLE_TWEETS[0], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
    
    for _ in range(NUM_BENCHMARK_RUNS * 5):
        with BenchmarkTimer("single_keybert") as timer:
            _ = kw_model.extract_keywords(SAMPLE_TWEETS[0], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
        single_times.append(timer.elapsed)
    
    results["single_text"] = format_results(single_times)
    print(f"  Single text: {results['single_text']['avg_ms']:.2f}ms avg")
    
    # Batch
    for n in [10, 50]:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            with BenchmarkTimer(f"keybert_{n}") as timer:
                for text in texts:
                    _ = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["throughput_per_sec"] = round(n / (stats["avg_ms"] / 1000), 1)
        results[f"batch_{n}"] = stats
        
        print(f"  {n:4} texts: {stats['avg_ms']:7.2f}ms avg | {stats['throughput_per_sec']:.1f} texts/sec")
    
    return results


def benchmark_deduplication():
    """Benchmark tweet deduplication pipeline."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Deduplication Pipeline")
    print("="*60)
    
    import re
    
    def deduplicate_tweets(df):
        original_count = len(df)
        df = df.drop_duplicates(subset=['text'], keep='first')
        exact_dupes = original_count - len(df)
        
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
        
        return df, {
            'original': original_count,
            'exact_duplicates': exact_dupes,
            'near_duplicates': near_dupes,
            'unique': len(df)
        }
    
    results = {}
    
    for n in [100, 500, 1000]:
        tweets = generate_test_data(n)
        for i in range(int(n * 0.2)):
            tweets.append(tweets[i % len(tweets)].copy())
        
        df = pd.DataFrame(tweets)
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            test_df = df.copy()
            with BenchmarkTimer(f"dedup_{n}") as timer:
                _, stats = deduplicate_tweets(test_df)
            times.append(timer.elapsed)
        
        benchmark_stats = format_results(times)
        results[f"{n}_tweets"] = benchmark_stats
        
        print(f"  {n:4} tweets: {benchmark_stats['avg_ms']:7.2f}ms avg")
    
    return results


def benchmark_tfidf_clustering():
    """Benchmark TF-IDF cluster keyword extraction."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: TF-IDF Cluster Keywords")
    print("="*60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    results = {}
    
    for n in [50, 100, 500]:
        texts = [SAMPLE_TWEETS[i % len(SAMPLE_TWEETS)] for i in range(n)]
        times = []
        
        for _ in range(NUM_BENCHMARK_RUNS):
            with BenchmarkTimer(f"tfidf_{n}") as timer:
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
                top_indices = mean_tfidf.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices]
            times.append(timer.elapsed)
        
        stats = format_results(times)
        results[f"{n}_tweets"] = stats
        
        print(f"  {n:4} tweets: {stats['avg_ms']:7.2f}ms avg")
    
    return results


def benchmark_full_pipeline():
    """Benchmark the complete tweet processing pipeline."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Full Pipeline (End-to-End)")
    print("="*60)
    
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from umap import UMAP
    
    # Load models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sent_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)
    
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    
    def get_sentiment(text):
        inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sent_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]
        labels = ['negative', 'neutral', 'positive']
        max_idx = torch.argmax(scores).item()
        return {'label': labels[max_idx], 'score': float(scores[2] - scores[0])}
    
    def get_emotion(text):
        result = emotion_pipe(text, truncation=True, max_length=512)
        return {'emotion': result[0]['label'], 'score': round(result[0]['score'], 3)}
    
    results = {}
    
    for n in [25, 50, 100]:
        tweets = generate_test_data(n)
        texts = [t["text"] for t in tweets]
        times = []
        
        for _ in range(min(NUM_BENCHMARK_RUNS, 3)):
            with BenchmarkTimer(f"pipeline_{n}") as timer:
                # Step 1: Embeddings
                embeddings = embedding_model.encode(texts, show_progress_bar=False)
                
                # Step 2: UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, n-1))
                coords = reducer.fit_transform(embeddings)
                
                # Step 3: Sentiment
                sentiments = [get_sentiment(text) for text in texts]
                
                # Step 4: Emotion
                emotions = [get_emotion(text) for text in texts]
            
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["per_tweet_ms"] = round(stats["avg_ms"] / n, 2)
        results[f"{n}_tweets"] = stats
        
        print(f"  {n:4} tweets: {stats['avg_ms']/1000:6.2f}s total | {stats['per_tweet_ms']:.2f}ms/tweet")
    
    return results


def benchmark_cache_operations():
    """Benchmark SQLite cache read/write performance at different table sizes."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: SQLite Cache Operations")
    print("="*60)
    
    import sqlite3
    from datetime import timedelta
    
    results = {}
    temp_db = os.path.join(os.path.dirname(__file__), "benchmark_cache_test.db")
    
    def setup_test_db(num_entries):
        """Populate cache with N entries."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                results_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON query_cache(topic)")
        cursor.execute("DELETE FROM query_cache")
        
        sample_result = {"topic": "test", "total_tweets": 50, "data": generate_test_data(50)}
        sample_json = json.dumps(sample_result)
        now = datetime.now()
        expires = now + timedelta(hours=6)
        
        for i in range(num_entries):
            cursor.execute("""
                INSERT INTO query_cache (topic, results_json, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, 0)
            """, (f"topic_{i}", sample_json, now.isoformat(), expires.isoformat()))
        conn.commit()
        conn.close()
        return sample_json
    
    def cache_lookup(topic):
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT results_json FROM query_cache WHERE topic = ?", (topic,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    for table_size in [10, 50, 100]:
        print(f"\n  Table Size: {table_size} entries")
        sample_json = setup_test_db(table_size)
        
        # Benchmark CACHE HIT
        hit_times = []
        for _ in range(NUM_BENCHMARK_RUNS * 20):
            with BenchmarkTimer("cache_hit") as timer:
                result = cache_lookup(f"topic_{table_size // 2}")
            hit_times.append(timer.elapsed)
        
        hit_stats = format_results(hit_times)
        print(f"    Cache HIT:  {hit_stats['avg_ms']:.3f}ms avg | {hit_stats['min_ms']:.3f}ms min")
        
        # Benchmark CACHE MISS
        miss_times = []
        for _ in range(NUM_BENCHMARK_RUNS * 20):
            with BenchmarkTimer("cache_miss") as timer:
                result = cache_lookup("nonexistent_topic_xyz")
            miss_times.append(timer.elapsed)
        
        miss_stats = format_results(miss_times)
        print(f"    Cache MISS: {miss_stats['avg_ms']:.3f}ms avg | {miss_stats['min_ms']:.3f}ms min")
        
        results[f"{table_size}_entries"] = {"hit": hit_stats, "miss": miss_stats}
    
    if os.path.exists(temp_db):
        os.remove(temp_db)
    
    return results


def benchmark_cache_with_json_parsing():
    """Benchmark cache lookup INCLUDING JSON parsing (real-world scenario)."""
    print("\n" + "="*60)
    print("📊 BENCHMARK: Cache + JSON Parsing (Real-World)")
    print("="*60)
    
    import sqlite3
    from datetime import timedelta
    
    results = {}
    temp_db = os.path.join(os.path.dirname(__file__), "benchmark_cache_json_test.db")
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)
    cursor.execute("DELETE FROM query_cache")
    
    for num_tweets in [25, 50, 100, 200]:
        sample_result = {"topic": "test", "total_tweets": num_tweets, "data": generate_test_data(num_tweets)}
        sample_json = json.dumps(sample_result)
        json_size_kb = len(sample_json) / 1024
        
        now = datetime.now()
        expires = now + timedelta(hours=6)
        
        cursor.execute("""
            INSERT OR REPLACE INTO query_cache (topic, results_json, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (f"topic_{num_tweets}", sample_json, now.isoformat(), expires.isoformat()))
        conn.commit()
        
        times = []
        for _ in range(NUM_BENCHMARK_RUNS * 20):
            with BenchmarkTimer("cache_full") as timer:
                cursor.execute("SELECT results_json FROM query_cache WHERE topic = ?", (f"topic_{num_tweets}",))
                row = cursor.fetchone()
                if row:
                    parsed = json.loads(row[0])
            times.append(timer.elapsed)
        
        stats = format_results(times)
        stats["json_size_kb"] = round(json_size_kb, 1)
        results[f"{num_tweets}_tweets"] = stats
        
        print(f"  {num_tweets:4} tweets ({json_size_kb:.1f}KB): {stats['avg_ms']:.2f}ms avg | {stats['min_ms']:.2f}ms min")
    
    conn.close()
    if os.path.exists(temp_db):
        os.remove(temp_db)
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def run_all_benchmarks():
    """Execute all benchmarks and generate report."""
    
    print("\n" + "="*60)
    print("  🚀 TweetScape Performance Benchmark Suite")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    all_results = {}
    
    try:
        all_results["embedding"] = benchmark_embedding_generation()
    except Exception as e:
        print(f"  ❌ Embedding benchmark failed: {e}")
    
    try:
        all_results["sentiment"] = benchmark_sentiment_analysis()
    except Exception as e:
        print(f"  ❌ Sentiment benchmark failed: {e}")
    
    try:
        all_results["emotion"] = benchmark_emotion_analysis()
    except Exception as e:
        print(f"  ❌ Emotion benchmark failed: {e}")
    
    try:
        all_results["umap"] = benchmark_umap_reduction()
    except Exception as e:
        print(f"  ❌ UMAP benchmark failed: {e}")
    
    try:
        all_results["keybert"] = benchmark_keybert_extraction()
    except Exception as e:
        print(f"  ❌ KeyBERT benchmark failed: {e}")
    
    try:
        all_results["deduplication"] = benchmark_deduplication()
    except Exception as e:
        print(f"  ❌ Deduplication benchmark failed: {e}")
    
    try:
        all_results["tfidf"] = benchmark_tfidf_clustering()
    except Exception as e:
        print(f"  ❌ TF-IDF benchmark failed: {e}")
    
    try:
        all_results["full_pipeline"] = benchmark_full_pipeline()
    except Exception as e:
        print(f"  ❌ Full pipeline benchmark failed: {e}")
    
    try:
        all_results["cache_operations"] = benchmark_cache_operations()
    except Exception as e:
        print(f"  ❌ Cache operations benchmark failed: {e}")
    
    try:
        all_results["cache_json"] = benchmark_cache_with_json_parsing()
    except Exception as e:
        print(f"  ❌ Cache JSON benchmark failed: {e}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("  📋 BENCHMARK SUMMARY (for Resume)")
    print("="*60)
    
    summary = []
    
    if "embedding" in all_results:
        throughput = all_results["embedding"].get("100_texts", {}).get("throughput_per_sec", "N/A")
        summary.append(f"• Embedding generation: ~{throughput} texts/sec (384-dim vectors)")
    
    if "sentiment" in all_results:
        single_ms = all_results["sentiment"].get("single_text", {}).get("avg_ms", "N/A")
        summary.append(f"• Sentiment analysis: ~{single_ms:.1f}ms per tweet (RoBERTa)")
    
    if "emotion" in all_results:
        single_ms = all_results["emotion"].get("single_text", {}).get("avg_ms", "N/A")
        summary.append(f"• Emotion detection: ~{single_ms:.1f}ms per tweet (7-class DistilRoBERTa)")
    
    if "umap" in all_results:
        time_200 = all_results["umap"].get("200_points", {}).get("avg_ms", "N/A")
        summary.append(f"• UMAP reduction: ~{time_200/1000:.2f}s for 200 points (384D → 2D)")
    
    if "full_pipeline" in all_results:
        per_tweet = all_results["full_pipeline"].get("100_tweets", {}).get("per_tweet_ms", "N/A")
        summary.append(f"• Full pipeline: ~{per_tweet}ms per tweet (end-to-end)")
    
    if "cache_operations" in all_results:
        cache_100 = all_results["cache_operations"].get("100_entries", {}).get("hit", {}).get("avg_ms", "N/A")
        summary.append(f"• SQLite cache (100 entries): ~{cache_100:.2f}ms lookup")
    
    if "cache_json" in all_results:
        cache_100_json = all_results["cache_json"].get("100_tweets", {}).get("avg_ms", "N/A")
        summary.append(f"• Cache + JSON parse (100 tweets): ~{cache_100_json:.2f}ms")
    
    for s in summary:
        print(s)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Full results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
