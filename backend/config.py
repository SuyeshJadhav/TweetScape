"""
Centralized configuration for TweetScape backend.
All paths, constants, and model names in one place.
"""
import os
import logging

# ============ PATHS ============
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# ============ SCRAPING ============
TWEETS_PER_QUERY = 10
MAX_SCROLL_ATTEMPTS = 10
COOKIES_FILE = os.path.join(BACKEND_DIR, "cookies.json")

# ============ MODELS ============
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# ============ UMAP ============
UMAP_COMPONENTS = 2
UMAP_RANDOM_STATE = 42

# ============ AGENT ============
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

# ============ LOGGING ============
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_LEVEL = logging.INFO

def setup_logging(name: str) -> logging.Logger:
    """Create a logger for a module."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger
