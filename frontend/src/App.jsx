import './index.css'
import SearchBar from './components/SearchBar'
import ClusterMap from './components/ClusterMap'
import TweetList from './components/TweetList'
import { useState } from 'react'
import Loading from './components/Loading'
import { clusterTopic } from './api/client'

function App() {
	const [loading, setLoading] = useState(false)
	const [clusterData, setClusterData] = useState(null)
	const [error, setError] = useState(null)

	const handleSearch = async (searchTopic, limit = 10) => {
		setLoading(true)
		setError(null)
		console.log('App: Starting full pipeline for:', searchTopic, 'limit:', limit)
		try {
			// Step 1: Scrape tweets
			console.log('Step 1: Starting scrape + cluster with expanded queries...')
			const clusterResponse = await clusterTopic(searchTopic)
			console.log('Cluster Response:', clusterResponse)

			// Step 2: Set data
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

				// Count by sentiment
				const sentimentCounts = {
					positive: data.filter(t => t.sentiment_score > 0.1).length,
					neutral: data.filter(t => Math.abs(t.sentiment_score) <= 0.1).length,
					negative: data.filter(t => t.sentiment_score < -0.1).length,
				}

				// Average sentiment score
				const avgScore = data.length > 0
					? (data.reduce((sum, t) => sum + (t.sentiment_score || 0), 0) / data.length).toFixed(2)
					: '--'

				// Polarization Score: variance of sentiment scores (0 = consensus, 1 = divided)
				let polarization = '--'
				if (data.length > 1) {
					const scores = data.map(t => t.sentiment_score || 0)
					const mean = scores.reduce((a, b) => a + b, 0) / scores.length
					const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length
					// Normalize: max variance for [-1,1] range is 1, so variance of 0.5+ is high
					polarization = Math.min(1, variance * 2).toFixed(2)
				}

				return (
					<div className="metrics-container">
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#f472b6' } }>{ data.length || '--' }</div>
							<div className="metric-label">Total Tweets</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#22c55e' } }>{ sentimentCounts.positive || '--' }</div>
							<div className="metric-label">😊 Positive</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#ef4444' } }>{ sentimentCounts.negative || '--' }</div>
							<div className="metric-label">😠 Negative</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: '#6b7280' } }>{ sentimentCounts.neutral || '--' }</div>
							<div className="metric-label">😐 Neutral</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: avgScore > 0 ? '#22c55e' : avgScore < 0 ? '#ef4444' : '#6b7280' } }>{ avgScore }</div>
							<div className="metric-label">Avg Score</div>
						</div>
						<div className="metric-card">
							<div className="metric-value" style={ { color: polarization > 0.5 ? '#ef4444' : polarization > 0.25 ? '#eab308' : '#22c55e' } }>{ polarization }</div>
							<div className="metric-label">⚡ Polarization</div>
						</div>
						{ clusterData?.dedup_stats && (
							<div className="metric-card">
								<div className="metric-value" style={ { color: '#a855f7' } }>
									{ clusterData.dedup_stats.original_count - clusterData.dedup_stats.unique_count }
								</div>
								<div className="metric-label">🤖 Duplicates Removed</div>
							</div>
						) }
					</div>
				)
			})() }

			{/* AI Summary */ }
			{ clusterData?.summary && (
				<div className="legend-container" style={ { marginTop: '1rem', background: '#f0fdf4', borderColor: '#22c55e' } }>
					<div className="legend-title">
						🤖 AI Summary
					</div>
					<p style={ { fontSize: '0.9rem', lineHeight: 1.6, margin: 0 } }>
						{ clusterData.summary }
					</p>
				</div>
			) }

			{/* Top Topics */ }
			{ clusterData?.topic_stats?.top_topics && (
				<div className="legend-container" style={ { marginTop: '1rem' } }>
					<div className="legend-title">
						🏷️ Top Topics
					</div>
					<div style={ { display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' } }>
						{ clusterData.topic_stats.top_topics.slice(0, 8).map(([topic, count], i) => (
							<span key={ i } style={ {
								padding: '4px 10px',
								background: '#1a1a1a',
								color: '#fff',
								borderRadius: '4px',
								fontSize: '0.75rem',
								fontFamily: "'Space Mono', monospace"
							} }>
								{ topic } ({ count })
							</span>
						)) }
					</div>
				</div>
			) }

			{/* Cluster Visualization */ }
			<ClusterMap
				data={ clusterData?.data || [] }
			/>

			{/* Tug of War Legend */ }
			<div className="legend-container">
				<div className="legend-title">
					⚔️ How to Read the Map
				</div>
				<div style={ { fontSize: '0.75rem', color: '#555', marginBottom: '0.75rem', lineHeight: 1.5 } }>
					<strong>Position (X-axis):</strong> Left = Negative, Right = Positive<br />
					<strong>Color:</strong> Emotion driving the opinion
				</div>
				<div className="legend-items" style={ { display: 'flex', flexWrap: 'wrap', gap: '0.5rem' } }>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#ef4444' } }></div>
						<div className="legend-text">Anger</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#22c55e' } }></div>
						<div className="legend-text">Joy</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#3b82f6' } }></div>
						<div className="legend-text">Sadness</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#6366f1' } }></div>
						<div className="legend-text">Fear</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#eab308' } }></div>
						<div className="legend-text">Surprise</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#a855f7' } }></div>
						<div className="legend-text">Disgust</div>
					</div>
					<div className="legend-item">
						<div className="legend-dot" style={ { backgroundColor: '#6b7280' } }></div>
						<div className="legend-text">Neutral</div>
					</div>
				</div>
				<div style={ {
					fontSize: '0.6rem',
					color: '#888',
					marginTop: '0.75rem',
					fontStyle: 'italic',
					lineHeight: 1.4
				} }>
					💡 "Dumbbell" shape = High polarization | Clustered center = Consensus
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
