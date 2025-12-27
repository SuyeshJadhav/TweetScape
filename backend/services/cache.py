"""
SQLite Query Cache Service
==========================
Caches processed pipeline results to avoid redundant scraping and ML inference.
"""

from config import CACHE_DB_PATH, CACHE_TTL_HOURS
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config import BACKEND_DIR, setup_logging

logger = setup_logging("cache")


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache_db():
    """Initialize cache database and table."""
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            hit_count INTEGER DEFAULT 0
        )
    """)
    
    # Index for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_topic ON query_cache(topic)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Cache database initialized at {CACHE_DB_PATH}")


def get_cached_result(topic: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached result for a topic if it exists and hasn't expired.
    
    Args:
        topic: The search topic to look up
        
    Returns:
        Cached result dict or None if not found/expired
    """
    init_cache_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    # Normalize topic for consistent cache keys
    normalized_topic = topic.strip().lower()
    
    cursor.execute("""
        SELECT id, results_json, expires_at, hit_count
        FROM query_cache
        WHERE topic = ?
    """, (normalized_topic,))
    
    row = cursor.fetchone()
    
    if row is None:
        conn.close()
        logger.info(f"Cache MISS for topic: {topic}")
        return None
    
    # Check expiration
    expires_at = datetime.fromisoformat(row['expires_at'])
    if datetime.now() > expires_at:
        # Expired - delete and return None
        cursor.execute("DELETE FROM query_cache WHERE id = ?", (row['id'],))
        conn.commit()
        conn.close()
        logger.info(f"Cache EXPIRED for topic: {topic}")
        return None
    
    # Update hit count
    cursor.execute("""
        UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = ?
    """, (row['id'],))
    conn.commit()
    
    result = json.loads(row['results_json'])
    result['_from_cache'] = True
    result['_cache_hits'] = row['hit_count'] + 1
    
    conn.close()
    logger.info(f"Cache HIT for topic: {topic} (hit #{row['hit_count'] + 1})")
    return result


def store_result(topic: str, data: Dict[str, Any], ttl_hours: int = CACHE_TTL_HOURS):
    """
    Store processed result in cache.
    
    Args:
        topic: The search topic
        data: The processed result to cache
        ttl_hours: Time-to-live in hours
    """
    init_cache_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    normalized_topic = topic.strip().lower()
    now = datetime.now()
    expires_at = now + timedelta(hours=ttl_hours)
    
    # Remove internal cache metadata before storing
    data_to_store = {k: v for k, v in data.items() if not k.startswith('_')}
    
    cursor.execute("""
        INSERT OR REPLACE INTO query_cache (topic, results_json, created_at, expires_at, hit_count)
        VALUES (?, ?, ?, ?, 0)
    """, (normalized_topic, json.dumps(data_to_store), now.isoformat(), expires_at.isoformat()))
    
    conn.commit()
    conn.close()
    logger.info(f"Cached result for topic: {topic} (expires in {ttl_hours}h)")


def clear_expired():
    """Remove all expired cache entries."""
    init_cache_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    cursor.execute("DELETE FROM query_cache WHERE expires_at < ?", (now,))
    deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    if deleted > 0:
        logger.info(f"Cleared {deleted} expired cache entries")
    return deleted


def clear_all():
    """Clear entire cache."""
    init_cache_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM query_cache")
    deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    logger.info(f"Cleared all {deleted} cache entries")
    return deleted


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    init_cache_db()
    conn = _get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as total FROM query_cache")
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT SUM(hit_count) as hits FROM query_cache")
    row = cursor.fetchone()
    hits = row['hits'] if row['hits'] else 0
    
    cursor.execute("""
        SELECT topic, hit_count, created_at, expires_at 
        FROM query_cache 
        ORDER BY hit_count DESC 
        LIMIT 5
    """)
    top_cached = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_cached_queries": total,
        "total_cache_hits": hits,
        "top_cached_topics": top_cached,
        "cache_db_path": CACHE_DB_PATH
    }


# Initialize on import
init_cache_db()
