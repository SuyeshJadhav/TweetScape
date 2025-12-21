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
 * Cluster tweets for a topic
 * @param {string} topic - The topic to cluster
 */
export const clusterTopic = async (topic) => {
	const response = await client.post(`/api/cluster/${topic}`)
	return response.data
}

export default client