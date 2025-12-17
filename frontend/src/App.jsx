import './index.css'
import SearchBar from './components/SearchBar'
import ClusterMap from './components/ClusterMap'
import TweetList from './components/TweetList'
import { useState } from 'react'
import Loading from './components/Loading'
import { analyzeTopic, clusterTopic, getClusterData } from './api/client'

function App() {
	const [topic, setTopic] = useState('')
	const [loading, setLoading] = useState(false)
	const [clusterData, setClusterData] = useState(null)
	const [error, setError] = useState(null)

	const handleSearch = async (searchTopic, limit = 10) => {
		setLoading(true)
		setError(null)
		console.log('App: Starting full pipeline for:', searchTopic, 'limit:', limit)
		try {
			// Step 1: Scrape tweets
			console.log('Step 1: Scraping tweets...')
			await analyzeTopic(searchTopic, limit)

			// Step 2: Cluster the scraped data
			console.log('Step 2: Clustering data...')
			const clusterResponse = await clusterTopic(searchTopic)
			console.log('Cluster Response:', clusterResponse)

			// Step 3: Set data
			setClusterData(clusterResponse)
		} catch (error) {
			console.error('Pipeline error:', error)
			setError(error.message)
		} finally {
			setLoading(false)
		}
	}


	return (
		<div className="app-container">
			{/* Header */ }
			<header className="header">
				<div className="logo">
					Twit<span className="logo-accent">Cluster</span>
				</div>
				<nav style={ { display: 'flex', gap: '1rem' } }>
					<button className="btn-secondary">DOCS</button>
				</nav>
			</header>

			{/* Search Section */ }
			<SearchBar
				onSearch={ handleSearch }
				loading={ loading }
			/>

			{/* Metrics */ }
			{ (() => {
				const data = clusterData?.data || []
				const positiveCount = data.filter(t => t.cluster === 2).length
				const negativeCount = data.filter(t => t.cluster === 0).length
				const neutralCount = data.filter(t => t.cluster === 1).length
				const avgScore = data.length > 0
					? (data.reduce((sum, t) => sum + (t.sentiment_score || 0), 0) / data.length).toFixed(2)
					: '--'

				return (
					<div className="metrics-container">
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#f472b6' } }>{ data.length || '--' }</div>
							<div className="metric-label">Total Tweets</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#22c55e' } }>{ positiveCount || '--' }</div>
							<div className="metric-label">😊 Positive</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#ef4444' } }>{ negativeCount || '--' }</div>
							<div className="metric-label">😠 Negative</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#6b7280' } }>{ neutralCount || '--' }</div>
							<div className="metric-label">😐 Neutral</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#1a1a1a' } }>{ avgScore || '--' }</div>
							<div className="metric-label">Avg Sentiment Score</div>
						</div>
					</div>
				)
			})() }

			{/* Cluster Visualization */ }
			<ClusterMap
				data={ clusterData?.data || [] }
			/>

			{/* Tribe Legend */ }
			<div className="legend-container">
				<div className="legend-title">
					📊 Sentiment Analysis
				</div>
				<div className="legend-items">
					<div className="legend-item">
						<div className="legend-dot cluster-0" style={ { backgroundColor: '#ef4444' } }></div>
						<div>
							<div className="legend-text">😠 Negative</div>
							<div className="legend-keywords">criticism, concerns, complaints</div>
						</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot cluster-1" style={ { backgroundColor: '#6b7280' } }></div>
						<div>
							<div className="legend-text">😐 Neutral</div>
							<div className="legend-keywords">news, facts, announcements</div>
						</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot cluster-2" style={ { backgroundColor: '#22c55e' } }></div>
						<div>
							<div className="legend-text">😊 Positive</div>
							<div className="legend-keywords">praise, excitement, enthusiasm</div>
						</div>
					</div>
				</div>
				<div style={ {
					fontSize: '0.6rem',
					color: '#888',
					marginTop: '0.75rem',
					fontStyle: 'italic',
					lineHeight: 1.4
				} }>
					⚠️ Note: Sarcasm and mockery may be misclassified
				</div>
			</div>

			{/* Tweet List */ }
			<TweetList
				tweets={ clusterData?.data || [] }
			/>

			{/* Loading State (conditionally render) */ }
			{ loading && <Loading /> }
		</div>
	)
}

export default App
