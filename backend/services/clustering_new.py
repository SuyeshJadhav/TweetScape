"""
Clustering module - UMAP dimensionality reduction and cluster assignment.
Focused only on spatial/clustering operations.
"""
import numpy as np
from config import setup_logging
from services.models import umap_reducer

logger = setup_logging("clustering")


def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce embeddings to 2D using UMAP for visualization.
    
    Args:
        embeddings: High-dimensional embeddings array
    
    Returns:
        2D coordinates array
    """
    logger.info(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]}D to 2D")
    coords = umap_reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete")
    return coords


def assign_clusters_by_sentiment(sentiments: list) -> list:
    """
    Assign cluster IDs based on sentiment labels.
    
    Clusters:
        0 = negative
        1 = neutral  
        2 = positive
    
    Args:
        sentiments: List of sentiment dicts with 'label' key
    
    Returns:
        List of cluster IDs
    """
    sentiment_to_cluster = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    clusters = [sentiment_to_cluster.get(s.get('label', 'neutral'), 1) for s in sentiments]
    
    # Log distribution
    cluster_counts = {0: 0, 1: 0, 2: 0}
    for c in clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    
    logger.info(f"Cluster distribution: negative={cluster_counts[0]}, neutral={cluster_counts[1]}, positive={cluster_counts[2]}")
    
    return clusters
