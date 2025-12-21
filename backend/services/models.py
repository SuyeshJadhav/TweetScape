"""
Model loading module - All ML models loaded here.
Separates heavy model initialization from business logic.
"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from umap import UMAP
from keybert import KeyBERT
from config import (
    EMBEDDING_MODEL, SENTIMENT_MODEL, EMOTION_MODEL,
    UMAP_COMPONENTS, UMAP_RANDOM_STATE, setup_logging
)

logger = setup_logging("models")

# ============ EMBEDDING MODEL ============
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ============ SENTIMENT MODEL ============
logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

# ============ EMOTION MODEL ============
logger.info(f"Loading emotion model: {EMOTION_MODEL}")
emotion_pipeline = pipeline("text-classification", model=EMOTION_MODEL)

# ============ UMAP ============
logger.info("Initializing UMAP reducer")
umap_reducer = UMAP(n_components=UMAP_COMPONENTS, random_state=UMAP_RANDOM_STATE)

# ============ KEYBERT ============
logger.info("Loading KeyBERT")
keybert_model = KeyBERT()

logger.info("All models loaded successfully!")
