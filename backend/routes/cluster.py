from fastapi import APIRouter
import json
import os
import sys

# Add services to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.clustering import process_topic_data

router = APIRouter()

@router.post("/cluster/{topic}")
def cluster_topic(topic: str):
	"""Run clustering on scraped tweets for a topic"""
	
	# Call clustering directly (not subprocess) to get all stats
	result = process_topic_data(topic)
	
	if result:
		return {
			"status": "success",
			"topic": result["topic"],
			"total_tweets": result["total_tweets"],
			"clusters": result["clusters"],
			"cluster_keywords": result["cluster_keywords"],
			"dedup_stats": result.get("dedup_stats"),
			"topic_stats": result.get("topic_stats"),
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
