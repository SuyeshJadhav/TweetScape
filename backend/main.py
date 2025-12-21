import asyncio
import sys

if sys.platform == "win32":
	asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.cluster import router as cluster_router
from routes.agent import router as agent_router

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)

app.include_router(analyze_router, prefix="/api")
app.include_router(cluster_router, prefix="/api")
app.include_router(agent_router, prefix="/agent", tags=["agent"])

@app.get("/")
def root():
	return {"status": "ok"}