function TweetList({ tweets }) {
	const displayTweets = tweets && tweets.length > 0 ? tweets : []

	const getSentimentLabel = (cluster) => {
		const labels = ['😠 Negative', '😐 Neutral', '😊 Positive']
		return labels[cluster] || 'Unknown'
	}

	const getSentimentColor = (cluster) => {
		const colors = ['#ef4444', '#6b7280', '#22c55e']
		return colors[cluster] || '#888'
	}

	const formatScore = (score) => {
		if (score === undefined || score === null) return ''
		const prefix = score > 0 ? '+' : ''
		return `${prefix}${score.toFixed(2)}`
	}

	return (
		<div className="tweets-container">
			<div className="tweets-header">
				<h2 className="tweets-title">📝 Tweet Feed</h2>
				<span className="tweets-count">{ displayTweets.length } tweets</span>
			</div>

			{ displayTweets.length === 0 ? (
				<div className="empty-state">
					<div className="empty-icon">🔍</div>
					<h3 className="empty-title">No Tweets Yet</h3>
					<p className="empty-text">Search for a topic to see tweets here.</p>
				</div>
			) : (
				<div className="tweets-grid">
					{ displayTweets.map((tweet, index) => (
						<div
							key={ index }
							className={ `tweet-card cluster-${tweet.cluster}` }
							style={ { borderLeftColor: getSentimentColor(tweet.cluster) } }
						>
							<div className="tweet-header">
								<span className="tweet-handle">{ tweet.handle?.split('\n')[0] }</span>
								<span
									className="tweet-cluster"
									style={ {
										backgroundColor: getSentimentColor(tweet.cluster),
										color: '#fff',
										padding: '2px 8px',
										borderRadius: '4px',
										fontSize: '0.7rem'
									} }
								>
									{ getSentimentLabel(tweet.cluster) }
								</span>
							</div>
							<p className="tweet-text">{ tweet.text }</p>
							<div className="tweet-footer" style={ { display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem' } }>
								<span
									className="tweet-score"
									style={ {
										fontWeight: 'bold',
										color: getSentimentColor(tweet.cluster)
									} }
								>
									{ formatScore(tweet.sentiment_score) }
								</span>
							</div>
						</div>
					)) }
				</div>
			) }
		</div>
	)
}

export default TweetList

