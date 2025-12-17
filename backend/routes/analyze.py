from models.schemas import AnalyzeRequest, AnalyzeResponse
from fastapi import APIRouter
import uuid
import subprocess
import json
import os
import sys

router = APIRouter()

jobs = {}

@router.post("/analyze")
def analyze(request: AnalyzeRequest):
	job_id = str(uuid.uuid4())
	
	jobs[job_id] = {
		"status": "running",
		"query": request.query,
		"result": None
	}

	# Run scraper as subprocess (avoids Windows event loop issues)
	backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	project_root = os.path.dirname(backend_dir)
	
	subprocess.run([
		sys.executable, "services/scraper.py",
		request.query,
		str(request.limit)
	], cwd=backend_dir)
	
	# Load result from saved file (data folder is in project root)
	data_file = os.path.join(project_root, "data", f"tweets_{request.query}.json")
	if os.path.exists(data_file):
		with open(data_file, "r", encoding="utf-8") as f:
			result = json.load(f)
	else:
		result = []
	
	jobs[job_id]["status"] = "completed"
	jobs[job_id]["result"] = result

	return AnalyzeResponse(
		job_id=job_id,
		status=jobs[job_id]["status"],
		result=jobs[job_id]["result"]
	)

@router.get("/jobs/{job_id}")
def get_job(job_id: str):
	if job_id not in jobs:
		return {
			"error": "Job not found"
		}
	return jobs[job_id]