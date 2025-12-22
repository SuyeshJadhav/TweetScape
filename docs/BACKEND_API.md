# Backend API Reference

Base URL: `http://localhost:8000`

## Overview

The TweetScape backend is built with **FastAPI**. It handles tweet scraping (via Playwright), semantic analysis, clustering dynamics, and agentic AI features (via Ollama).

## Endpoints

### 1. Cluster Topic

**Trigger the full analysis pipeline:** Scrapes tweets, expands queries, analyzes sentiment/emotion, and generates clusters.

- **URL**: `/api/cluster/{topic}`
- **Method**: `POST`
- **Path Parameters**:
    - `topic` (string): The search topic (e.g., "iPhone 16").

**Process Flow:**
1. **Agent Expansion**: `AI Agent` -> Generates 5 related search queries.
2. **Scraping**: `Playwright` -> Scrapes ~10 tweets per query (deduplicated).
3. **Analysis**:
    - **Embeddings**: `all-MiniLM-L6-v2`
    - **Sentiment**: `twitter-roberta-base-sentiment`
    - **Emotion**: `emotion-english-distilroberta-base`
    - **Topics**: `KeyBERT`
4. **Clustering**: `UMAP` reduction -> Custom density clustering.
5. **Summarization**: `AI Agent` -> Generates executive summary.

**Response Schema:**
```json
{
  "status": "success",
  "topic": "iPhone 16",
  "total_tweets": 45,
  "clusters": 3,
  "cluster_keywords": {
    "0": ["battery", "camera", "zoom"],
    "1": ["price", "expensive", "worth"],
    "2": ["design", "colors", "titanium"]
  },
  "dedup_stats": {
    "original": 60,
    "unique": 45
  },
  "summary": "Public sentiment is generally positive regarding the new camera features...",
  "data": [
    {
      "handle": "@techreviewer",
      "text": "The zoom on the iPhone 16 is insane! 📸",
      "sentiment": "positive",
      "sentiment_score": 0.9,
      "emotion": "joy",
      "emotion_score": 0.85,
      "x": 12.5,
      "y": 4.2
    }
  ]
}
```

---

### 2. Get Cached Cluster

**Retrieve previously analyzed data** without re-scraping.

- **URL**: `/api/cluster/{topic}`
- **Method**: `GET`
- **Path Parameters**:
    - `topic` (string): The search topic.

**Response**: Same as `POST /api/cluster/{topic}`.

**Errors**:
- `404 Not Found`: If no data exists for the topic.

---

### 3. Expand Query (Agent)

**Ask the AI agent for related search terms.**

- **URL**: `/agent/expand/{topic}`
- **Method**: `POST`
- **Path Parameters**:
    - `topic` (string): Base topic.

**Response Schema:**
```json
{
  "queries": [
    "iPhone 16",
    "iPhone 16 pro max camera test",
    "iPhone 16 battery life review",
    "iPhone 16 vs 15 pro",
    "iPhone 16 heating issues"
  ]
}
```

---

### 4. Summarize Topic (Agent)

**Generate a summary from an existing cluster dataset.**

- **URL**: `/agent/summarize/{topic}`
- **Method**: `POST`
- **Path Parameters**:
    - `topic` (string): Base topic.
- **Body** (JSON):
    - `cluster_data` (dict): The full data object returned by the cluster endpoint.

**Response Schema:**
```json
{
  "summary": "Users are excited about the camera upgrades but concerned about the price increase..."
}
```

## Data Models

### Tweet Object

| Field | Type | Description |
|-------|------|-------------|
| `handle` | string | Twitter handle (e.g., "@user") |
| `text` | string | Tweet content |
| `sentiment` | string | `positive`, `negative`, or `neutral` |
| `sentiment_score` | float | Score from -1.0 to 1.0 |
| `emotion` | string | `joy`, `anger`, `sadness`, `fear`, `disgust`, `surprise`, `neutral` |
| `x` | float | UMAP coordinate X |
| `y` | float | UMAP coordinate Y |
| `cluster` | int | Cluster ID assignment |

## Configuration

Settings are managed in `backend/config.py`:

- **Models**:
    - `EMBEDDING_MODEL`
    - `SENTIMENT_MODEL`
    - `EMOTION_MODEL`
- **Agent**:
    - `OLLAMA_URL`: Default `http://localhost:11434/api/generate`
    - `OLLAMA_MODEL`: Default `gemma3:4b`
