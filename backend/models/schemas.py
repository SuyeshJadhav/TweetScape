from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
	mode: str
	query: str
	limit: int

class AnalyzeResponse(BaseModel):
	job_id: str
	status: str
	result: list