from fastapi import APIRouter
import subprocess
import json
import os
import sys

router = APIRouter()

@router.post("/cluster/{topic}")
def cluster_topic(topic: str):
	"""Run clustering on scraped tweets for a topic"""
	
	backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	project_root = os.path.dirname(backend_dir)
	
	# Run clustering as subprocess
	subprocess.run([
		sys.executable, "services/clustering.py",
		topic
	], cwd=backend_dir)
	
	# Load result from saved file
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
