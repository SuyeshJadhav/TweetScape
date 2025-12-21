import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"

def chat(prompt):
	"""Send prompt to Gemma3 and return response"""

	# 1. Create payload dictionary
	payload = {
		"model": MODEL,
		"prompt": prompt,
		"stream": False,
	}

	# 2. Make POST request to Ollama
	response = requests.post(OLLAMA_URL, json=payload)

	# 3. Check response status
	if response.status_code != 200:
		return None

	# 4. Return response data
	return response.json()["response"]

def expand_query(topic):
	"""Generate related search queries for broader tweet coverage."""

	# 1. Create a prompt asking for 5 related queries
	prompt = f"""
	You are a search query expander.
	Given a topic, generate 5 related search queries to find more tweets about this topic.

	Topic: {topic}

	Output ONLY 5 search queries, one per line. No numbering, no explanation.
	"""

	# 2. Call chat() with this prompt
	response = chat(prompt)

	# 3. Split response by newlines to get list of queries
	queries = response.strip().split("\n")

	# 4. Clean up (remove empty lines, strip whitespace)
	queries = [q.strip() for q in queries if q.strip()]

	# 5. Return list of queries 
	return [topic] + queries[:5]

def summarize_tweets(topic, stats, sample_tweets):
	"""Generate a summary of tweet analysis.
	
	Args:
		topic: The search topic
		stats: Dict with positive%, negative%, neutral%, emotions, topics
		sample_tweets: List of 5-10 sample tweet texts
	"""

	# 1. Format sample tweets as numbered list
	tweets_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(sample_tweets[:5])])

	# 2. Create prompt
	prompt = f"""Analyze tweets about "{topic}" and provide a 2-3 sentence summary.
	
	Stats:
	- Positive: {stats['positive']}%
	- Negative: {stats['negative']}%
	- Neutral: {stats['neutral']}%
	- Top emotions: {stats.get('emotions', 'N/A')}

	Sample tweets:
	{tweets_text}

	Write a concise summary of public opinion.
	"""
	# 3. Call chat() and return response
	return chat(prompt)