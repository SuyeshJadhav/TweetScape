import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize local embedding model (free, no API key needed)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading sentiment model...")
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

def get_sentiment(text):
	"""Analyze sentiment using RoBERTa model. Returns score from -1 (negative) to +1 (positive)."""
	inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
	with torch.no_grad():
		outputs = sentiment_model(**inputs)

	# Get probablitites: [negative, neutral, positive]
	probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

	# Calculate score: positive - negative (range: -1 to +1)
	score = probs[2] - probs[0]

	# Determine label
	if score > 0.1:
		label = "positive"
	elif score < -0.1:
		label = "negative"
	else:
		label = "neutral"
	
	return {
		"score": round(score, 3),
		"label": label,
		"confidence": round(max(probs), 3)
	}

def get_embeddings(texts):
	"""Generate embeddings locally using sentence-transformers."""
	print(f"Generating embeddings for {len(texts)} tweets...")
	embeddings = model.encode(texts, show_progress_bar=True)
	return embeddings

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

	# 3. Generate Embeddings
	embeddings = get_embeddings(df["text"].tolist())
	matrix = np.array(embeddings)

	# 4. Dimensionality Reduction (t-SNE for visualization)
	n_samples = len(df)
	perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

	print("Mapping high-dimensional space to 2D...")
	tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='random')
	vis_dims = tsne.fit_transform(matrix)

	df['x'] = vis_dims[:, 0].tolist()
	df['y'] = vis_dims[:, 1].tolist()

	# 5. Sentiment Analysis
	print("Analyzing sentiment...")
	sentiments = [get_sentiment(text) for text in df['text'].tolist()]
	df['sentiment_score'] = [s['score'] for s in sentiments]
	df['sentiment_label'] = [s['label'] for s in sentiments]

	# Map labels to cluster IDs for visualization
	label_to_cluster = {
		'negative': 0,
		'neutral': 1,
		'positive': 2
	}
	df['cluster'] = [label_to_cluster[s['label']] for s in sentiments]

	# 6. Get keywords for each cluster
	print("Extracting tribe keywords...")
	cluster_keywords = get_cluster_keywords(df)

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
		"data": output_data
	}

if __name__ == "__main__":
	import sys
	if len(sys.argv) >= 2:
		topic = sys.argv[1]
		result = process_topic_data(topic)
		if result:
			print(f"Processed {result['total_tweets']} tweets into {result['clusters']} clusters")