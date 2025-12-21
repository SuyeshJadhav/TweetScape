"""Agent service that orchestrates query expansion and summarization."""

from services.ollama_client import expand_query, summarize_tweets

def get_expanded_queries(topic):
	"""Get expanded search queries for a topic.
	
	Args:
		topic: Original search query
	
	Returns:
		List of search queries (original + expanded)
	"""
	# Call expand_query(topic)
	return expand_query(topic)

def get_summary(topic, cluster_data):
	"""Generate summary from clustered tweet data.
	
	Args:
		topic: Search topic
		cluster_data: Dict with 'data' (tweets), counts, etc.
	
	Returns:
		Summary string
	"""
	# 1. Get tweet data
	tweets = cluster_data.get('data', [])

	if not tweets:
		return "No tweets to summarize."
	
	# 2. Calculate stats from tweet data
	total = len(tweets)
	positive = len([t for t in tweets if t.get('sentiment') == 'positive'])
	negative = len([t for t in tweets if t.get('sentiment') == 'negative'])
	neutral = total - positive - negative

	# Convert to percentages
	stats = {
		'positive': round(positive /total * 100) if total > 0 else 0,
		'negative': round(negative /total * 100) if total > 0 else 0,
		'neutral': round(neutral /total * 100) if total > 0 else 0,
		'emotions': ', '.join(set([t.get('emotion', '') for t in tweets[:10]]))
	}

	# 3. Get sample tweets (text only)
	sample_tweets = [t.get('text', '')[:200] for t in tweets[:5]]

	# 4. Call summarize_tweets() and return
	return summarize_tweets(topic, stats, sample_tweets)