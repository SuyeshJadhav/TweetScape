import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import re
from keybert import KeyBERT

# Initialize local embedding model (free, no API key needed)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize sentiment model
print("Loading sentiment model...")
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Initialize UMAP
print("Loading UMAP...")
reducer = UMAP(n_components=2, random_state=42)

print("Loading emotion model...")
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_model = pipeline("text-classification", model=emotion_model_name)

# Initialize KeyBERT
print("Loading KeyBERT...")
keybert = KeyBERT()

def get_sentiment(text):
    """Analyze sentiment using RoBERTa. Returns score from -1 (negative) to +1 (positive)."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    score = probs[2] - probs[0]  # positive - negative
    
    if score > 0.1:
        label = "positive"
    elif score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    
    return {
        "score": round(score, 3),
        "label": label
    }

def get_emotion(text):
	"""Analyze emotion using j-hartmann fine tuned model."""
	result = emotion_model(text, truncation=True, max_length=512)
	return {
		"emotion": result[0]['label'],
		"score": round(result[0]['score'], 3)
	}

def get_embeddings(texts):
	"""Generate embeddings locally using sentence-transformers."""
	print(f"Generating embeddings for {len(texts)} tweets...")
	embeddings = model.encode(texts, show_progress_bar=True)
	return embeddings

def deduplicate_tweets(df):
	"""Remove duplicate and near-duplicate tweets (bot decision)"""
	original_count = len(df)

	# 1. Exact duplicates: same text
	df = df.drop_duplicates(subset=['text'], keep='first')
	exact_dupes = original_count - len(df)

	# 2. Near-duplicates: normalize text and check again
	# Remove URLs, mentions, extra spaces for comparison
	df['text_normalized'] = df['text'].apply(lambda x: re.sub(r'http\S+|@\w+|#\w+', '', x.lower().strip()))
	df['text_normalized'] = df['text_normalized'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

	# Remove near duplicates
	before_near = len(df)
	df = df.drop_duplicates(subset=['text_normalized'], keep='first')
	near_dupes = before_near - len(df)

	# clean up temporary column
	df = df.drop(columns=['text_normalized'])

	print(f"Deduplication: Removes {exact_dupes} exact + {near_dupes} near-duplicates")
	print(f"Unique tweets: {len(df)} / {original_count} ({100*len(df)//original_count}%)")

	return df, {
		'original_count': original_count,
		'exact_duplicates': exact_dupes,
		'near_duplicates': near_dupes,
		'unique_count': len(df)
	}

def extract_topics(df, keybert):
	"""Extract topics/keywords for each tweet using KeyBERT"""
	print("Extracting topics...")
	topics = []
	for text in df['text'].tolist():
		try:
			# Extract top 2 keywords per tweet
			keywords = keybert.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
			# Get just the keyword strings
			topic_list = [kw[0] for kw in keywords] if keywords else []
			topics.append(topic_list)
		except:
			topics.append([])

	df['topics'] = topics
	df['primary_topic'] = [t[0] if t else 'unknown' for t in topics]

	# Count topic frequency
	all_topics = [topic for topics_list in topics for topic in topics_list]
	topic_counts = {}
	for topic in all_topics:
		topic_counts[topic] = topic_counts.get(topic, 0) + 1
	
	# Get top 10 topics
	top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
	print(f"Top topics: {[t[0] for t in top_topics[:5]]}")

	return df, {'top_topics': top_topics}

def get_cluster_keywords(df, n_keywords=3):
	"""Extract top keywords for each cluster using TF-IDF."""
	cluster_keywords = {}
	
	for cluster_id in df['cluster'].unique():
		cluster_texts = df[df['cluster'] == cluster_id]['text'].tolist()
		
		if len(cluster_texts) < 2:
			cluster_keywords[int(cluster_id)] = []
			continue
			
		# TF-IDF to find distinctive words
		vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
		try:
			tfidf_matrix = vectorizer.fit_transform(cluster_texts)
			feature_names = vectorizer.get_feature_names_out()
			
			# Get top keywords by mean TF-IDF score
			mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
			top_indices = mean_tfidf.argsort()[-n_keywords:][::-1]
			keywords = [feature_names[i] for i in top_indices]
			cluster_keywords[int(cluster_id)] = keywords
		except:
			cluster_keywords[int(cluster_id)] = []
	
	return cluster_keywords

def process_topic_data(topic):
	"""Main pipeline: Load -> Embed -> Reduce -> Cluster -> Save"""
	
	# Get project root (data folder is in project root)
	project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	data_dir = os.path.join(project_root, "data")
	
	# Sanitize topic for filename (replace spaces with underscores)
	safe_topic = topic.replace(" ", "_")
	input_file = os.path.join(data_dir, f"tweets_{safe_topic}.json")
	output_file = os.path.join(data_dir, f"clustered_{safe_topic}.json")

	if not os.path.exists(input_file):
		print(f"File not found: {input_file}")
		return None
	
	# 1. Load Data
	with open(input_file, "r", encoding="utf-8") as f:
		raw_data = json.load(f)
	
	if len(raw_data) < 3:
		print("Not enough tweets for clustering (need at least 3)")
		return None
	
	# 2. Create DataFrame
	df = pd.DataFrame(raw_data)
	print(f"Loaded {len(df)} tweets.")

	df, dedup_stats = deduplicate_tweets(df)

	if len(df) < 3:
		print("Not enough tweets after deduplication (need at least 3)")
		return None

	# 3. Generate Embeddings
	embeddings = get_embeddings(df["text"].tolist())
	matrix = np.array(embeddings)

	# 4. Dimensionality Reduction (UMAP for visualization)
	print("Mapping high-dimensional space to 2D...")
	vis_dims = reducer.fit_transform(matrix)

	df['x'] = vis_dims[:, 0].tolist()
	df['y'] = vis_dims[:, 1].tolist()

	# 5. Sentiment and Emotion Analysis
	print("Analyzing sentiment...")
	sentiments = [get_sentiment(text) for text in df['text'].tolist()]
	df['sentiment'] = [s['label'] for s in sentiments]
	df['sentiment_score'] = [s['score'] for s in sentiments]

	print("Analyzing emotion...")
	emotions = [get_emotion(text) for text in df['text'].tolist()]
	df['emotion'] = [e['emotion'] for e in emotions]
	df['emotion_score'] = [e['score'] for e in emotions]

	# Map sentiment to cluster IDs (3 clusters: negative, neutral, positive)
	sentiment_to_cluster = {
		'negative': 0,
		'neutral': 1,
		'positive': 2
	}
	df['cluster'] = [sentiment_to_cluster.get(s['label'], 1) for s in sentiments]

	# 6. Get keywords for each cluster and topic
	print("Extracting tribe keywords...")
	cluster_keywords = get_cluster_keywords(df)
	df, topic_stats = extract_topics(df, keybert)

	# 7. Save clustered data
	output_data = df.to_dict(orient="records")

	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(output_data, f, indent=4, ensure_ascii=False)
	
	print(f"Success! Clustered data saved to {output_file}")
	
	# Return result for API
	return {
		"topic": topic,
		"total_tweets": len(df),
		"clusters": 3,
		"cluster_keywords": cluster_keywords,
		"dedup_stats": dedup_stats,
		"topic_stats": topic_stats,
		"data": output_data
	}

if __name__ == "__main__":
	import sys
	if len(sys.argv) >= 2:
		topic = sys.argv[1]
		result = process_topic_data(topic)
		if result:
			print(f"Processed {result['total_tweets']} tweets into {result['clusters']} clusters")