import axios from 'axios'

// Base URL for FastAPI backend
const API_BASE = 'http://localhost:8000'

const client = axios.create({
	baseURL: API_BASE,
	headers: {
		'Content-Type': 'application/json',
	},
})

/**
 * Analyze a topic - scrapes tweets
 * @param {string} topic - The topic to search
 * @param {number} limit - Number of tweets to scrape
 */
export const analyzeTopic = async (topic, limit = 10) => {
	const response = await client.post('/api/analyze', {
		mode: 'topic',
		query: topic,
		limit: limit,
	})
	return response.data
}

/**
 * Cluster tweets for a topic
 * @param {string} topic - The topic to cluster
 */
export const clusterTopic = async (topic) => {
	const response = await client.post(`/api/cluster/${topic}`)
	return response.data
}

/**
 * Get existing cluster data
 * @param {string} topic - The topic to get data for
 */
export const getClusterData = async (topic) => {
	const response = await client.get(`/api/cluster/${topic}`)
	return response.data
}

export default client