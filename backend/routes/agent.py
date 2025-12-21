from fastapi import APIRouter, Body

router = APIRouter()

@router.post("/expand/{topic}")
def expand_topic(topic: str):
	"""Expand a search query into related queries."""
	from services.agent import get_expanded_queries
	result = get_expanded_queries(topic)
	return {"queries": result}

@router.post("/summarize/{topic}")
def summarize_topic(topic: str, cluster_data: dict = Body(...)):
	"""Generate a summary from clustered tweet data."""
	from services.agent import get_summary
	result = get_summary(topic, cluster_data)
	return {"summary": result}