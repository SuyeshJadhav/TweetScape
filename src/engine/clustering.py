import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Initialize local embedding model (free, no API key needed)
# This will download ~90MB model on first run
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
	"""
	Generate embeddings locally using sentence-transformers (FREE).
	"""
	print(f"Generating embeddings for {len(texts)} tweets...")
	embeddings = model.encode(texts, show_progress_bar=True)
	return embeddings

def process_topic_data(topic):
	"""
	Main pipeline: Load -> Embed -> Reduce -> Cluster -> Save
	"""
	input_file = f"data/tweets_{topic}.json"
	output_file = f"data/clustered_{topic}.json"

	if not os.path.exists(input_file):
		print(f"File not found: {input_file}")
		return
	
	# 1. Load Data
	with open(input_file, "r", encoding="utf-8") as f:
		raw_data = json.load(f)
	
	# 2. Simple Cleaning (Remove very short tweets or duplicates)
	df = pd.DataFrame(raw_data)
	print(f"Loaded {len(df)} tweets.")

	# 3. Generate Embeddings
	# This is the "Semantic" step
	embeddings = get_embeddings(df["text"].tolist())
	matrix = np.array(embeddings)

	# 4. Dimensionality Reduction (The Map Marker)
	# We use t-SNE for organic looking clusters.
	# Perplexity must be < number of samples.
	n_samples = len(df)
	perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

	print("Mapping high-dimensional space to 2D...")
	tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='random')
	vis_dims = tsne.fit_transform(matrix)

	df['x'] = vis_dims[:, 0]
	df['y'] = vis_dims[:, 1]

	# 5. Clustering (The Tribe Finder)
	# We force 3 clusters for now (Pro, Anti, Neutral/Spam)
	print("Identifying tribes...")
	kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
	df['cluster'] = kmeans.fit_predict(matrix)

	# 6. Save for Dashboard
	# Convert back to list of dicts for JSON saving
	output_data = df.to_dict(orient="records")

	with open(output_file, "w", encoding="utf-8") as f:
		json.dump(output_data, f, indent=4, ensure_ascii=False)
	
	print(f"Success! Clustered data saved to {output_file}")

if __name__ == "__main__":
	# Test with your current topic
	process_topic_data("ChatGPT")