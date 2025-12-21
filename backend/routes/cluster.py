from fastapi import APIRouter
import json
import os
import sys
import subprocess

# Add services to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.pipeline import process_topic_data
from services.agent import get_expanded_queries, get_summary

router = APIRouter()

@router.post("/cluster/{topic}")
def cluster_topic(topic: str):
	"""Run clustering on scraped tweets for a topic"""

	# Define paths
	backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	project_root = os.path.dirname(backend_dir)
	data_dir = os.path.join(project_root, "data")

	# 1. Get expanded queries for topic
	expanded_queries = get_expanded_queries(topic)
	print(f"Expanded queries to {len(expanded_queries)}")

	# 2. Scrape tweets for each query
	all_tweets = []
	for query in expanded_queries:
		# Run scraper subprocess for this query
		subprocess.run([
			sys.executable, "services/scraper.py",
			query,
			"10"
		], cwd = backend_dir)

		# Load results
		safe_query = query.replace(" ", "_")
		data_file = os.path.join(data_dir, f"tweets_{safe_query}.json")
		
		if os.path.exists(data_file):
			with open(data_file, "r", encoding="utf-8") as f:
				tweets = json.load(f)
				all_tweets.extend(tweets)
		else:
			print(f"No tweets found for query: {query}")
	
	# 3. Save combined tweets to original topic file
	safe_topic = topic.replace(" ", "_")
	combined_file = os.path.join(data_dir, f"tweets_{safe_topic}.json")
	with open(combined_file, "w", encoding="utf-8") as f:
		json.dump(all_tweets, f, indent=4, ensure_ascii=False)
	
	# 4. Cleanup intermediate files (keep only combined file)
	for query in expanded_queries:
		safe_query = query.replace(" ", "_")
		if safe_query != safe_topic:  # Don't delete the combined file
			intermediate_file = os.path.join(data_dir, f"tweets_{safe_query}.json")
			if os.path.exists(intermediate_file):
				os.remove(intermediate_file)
				print(f"Cleaned up: tweets_{safe_query}.json")
	
	print(f"Combined {len(all_tweets)} tweets from {len(expanded_queries)} queries")
	
	# 4. Now run clustering (existing code)
	result = process_topic_data(topic)

	# 5. Generate AI summary
	summary = get_summary(topic, result) if result else None

	if result:
		return {
			"status": "success",
			"topic": result["topic"],
			"total_tweets": result["total_tweets"],
			"clusters": result["clusters"],
			"cluster_keywords": result["cluster_keywords"],
			"dedup_stats": result.get("dedup_stats"),
			"topic_stats": result.get("topic_stats"),
			"summary": summary,
			"data": result["data"]
		}
	else:
		return {
			"status": "error",
			"message": f"Failed to cluster topic: {topic}"
		}

@router.get("/cluster/{topic}")
def get_cluster(topic: str):
	"""Get existing clustered data for a topic"""
	
	project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
	safe_topic = topic.replace(" ", "_")
	data_file = os.path.join(project_root, "data", f"clustered_{safe_topic}.json")
	
	if os.path.exists(data_file):
		with open(data_file, "r", encoding="utf-8") as f:
			result = json.load(f)
		return {
			"status": "success",
			"topic": topic,
			"total_tweets": len(result),
			"data": result
		}
	else:
		return {
			"status": "error",
			"message": f"No clustered data found for topic: {topic}"
		}
