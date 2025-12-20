import * as d3 from 'd3'
import { useRef, useState, useEffect } from 'react'

function ClusterMap({ data, onPointClick }) {
	const svgRef = useRef()
	const [tooltipData, setTooltipData] = useState(null)
	const hasData = data && data.length > 0

	// Emotion colors (primary fill in Tug of War mode)
	const emotionColors = {
		'anger': '#ef4444',    // red
		'disgust': '#a855f7',  // purple
		'fear': '#6366f1',     // indigo
		'joy': '#22c55e',      // green
		'neutral': '#6b7280',  // gray
		'sadness': '#3b82f6',  // blue
		'surprise': '#eab308'  // yellow
	}

	const emotionEmojis = {
		'anger': '😠',
		'disgust': '🤢',
		'fear': '😨',
		'joy': '😊',
		'neutral': '😐',
		'sadness': '😢',
		'surprise': '😲'
	}

	useEffect(() => {
		if (!hasData || !svgRef.current) return

		const width = 780
		const height = 600
		const padding = 80

		// Adaptive node size based on count
		const nodeCount = data.length
		const nodeRadius = Math.max(6, Math.min(14, 180 / Math.sqrt(nodeCount)))

		// "Tug of War" layout: X = sentiment score, Y = random spread
		const nodes = data.map((tweet, i) => {
			// X position: sentiment score (-1 to +1) maps to (padding to width-padding)
			const sentimentScore = tweet.sentiment_score || 0
			const xPos = padding + ((sentimentScore + 1) / 2) * (width - 2 * padding)

			// Y position: random spread within bounds
			const yPos = height * 0.15 + Math.random() * height * 0.7

			return {
				id: i,
				...tweet,
				x: xPos,
				y: yPos,
				targetX: xPos  // Store target for force simulation
			}
		})

		// Force simulation - keeps nodes separated but pulls toward their X target
		const simulation = d3.forceSimulation(nodes)
			.force('charge', d3.forceManyBody().strength(-8))
			.force('collide', d3.forceCollide(nodeRadius + 1))
			.force('x', d3.forceX(d => d.targetX).strength(0.3))
			.force('y', d3.forceY(height / 2).strength(0.02))
			.alphaDecay(0.03)
			.velocityDecay(0.3)

		const svg = d3.select(svgRef.current)
		svg.selectAll('*').remove()

		// Background gradient (dark)
		const defs = svg.append('defs')

		// Axis labels for Tug of War
		svg.append('text')
			.attr('x', padding)
			.attr('y', height - 20)
			.attr('text-anchor', 'middle')
			.attr('fill', '#ef4444')
			.attr('font-size', '14px')
			.attr('font-family', 'Space Mono, monospace')
			.attr('font-weight', 'bold')
			.text('← NEGATIVE')

		svg.append('text')
			.attr('x', width - padding)
			.attr('y', height - 20)
			.attr('text-anchor', 'middle')
			.attr('fill', '#22c55e')
			.attr('font-size', '14px')
			.attr('font-family', 'Space Mono, monospace')
			.attr('font-weight', 'bold')
			.text('POSITIVE →')

		svg.append('text')
			.attr('x', width / 2)
			.attr('y', height - 20)
			.attr('text-anchor', 'middle')
			.attr('fill', '#6b7280')
			.attr('font-size', '12px')
			.attr('font-family', 'Space Mono, monospace')
			.text('NEUTRAL')

		// Center line (neutral zone)
		svg.append('line')
			.attr('x1', width / 2)
			.attr('y1', 40)
			.attr('x2', width / 2)
			.attr('y2', height - 50)
			.attr('stroke', '#333')
			.attr('stroke-width', 1)
			.attr('stroke-dasharray', '4,4')

		// Green glow for strong positive
		const glowGreen = defs.append('filter').attr('id', 'glow-green')
		glowGreen.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur')
		glowGreen.append('feFlood').attr('flood-color', '#22c55e').attr('flood-opacity', '0.8')
		glowGreen.append('feComposite').attr('in2', 'blur').attr('operator', 'in')
		glowGreen.append('feMerge')
			.selectAll('feMergeNode')
			.data(['', 'SourceGraphic'])
			.enter()
			.append('feMergeNode')
			.attr('in', d => d || null)

		// Red glow for strong negative
		const glowRed = defs.append('filter').attr('id', 'glow-red')
		glowRed.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur')
		glowRed.append('feFlood').attr('flood-color', '#ef4444').attr('flood-opacity', '0.8')
		glowRed.append('feComposite').attr('in2', 'blur').attr('operator', 'in')
		glowRed.append('feMerge')
			.selectAll('feMergeNode')
			.data(['', 'SourceGraphic'])
			.enter()
			.append('feMergeNode')
			.attr('in', d => d || null)

		const nodeGroup = svg.append('g').attr('class', 'nodes')

		const circles = nodeGroup.selectAll('circle')
			.data(nodes)
			.enter()
			.append('circle')
			.attr('r', nodeRadius)
			.attr('fill', d => {
				// Fill color = Emotion
				const color = emotionColors[d.emotion] || '#6b7280'
				const intensity = d.emotion_score || 0.7
				const opacity = 0.5 + intensity * 0.5
				const r = parseInt(color.slice(1, 3), 16)
				const g = parseInt(color.slice(3, 5), 16)
				const b = parseInt(color.slice(5, 7), 16)
				return `rgba(${r}, ${g}, ${b}, ${opacity})`
			})
			.attr('stroke', d => {
				// Stroke color = Sentiment (red/gray/green border)
				if (d.sentiment_score > 0.1) return '#22c55e'
				if (d.sentiment_score < -0.1) return '#ef4444'
				return '#6b7280'
			})
			.attr('stroke-width', 2)
			.attr('filter', d => {
				// Strong sentiment gets glow
				if (Math.abs(d.sentiment_score) > 0.7) {
					return d.sentiment_score > 0 ? 'url(#glow-green)' : 'url(#glow-red)'
				}
				return null
			})
			.style('cursor', 'pointer')
			.on('mouseover', function (event, d) {
				d3.select(this).attr('r', nodeRadius * 1.5).attr('stroke-width', 3)
				setTooltipData(d)
			})
			.on('mouseout', function (event, d) {
				d3.select(this).attr('r', nodeRadius).attr('stroke-width', 2)
				setTooltipData(null)
			})
			.call(d3.drag()
				.on('start', (event, d) => {
					if (!event.active) simulation.alphaTarget(0.3).restart()
					d.fx = d.x
					d.fy = d.y
				})
				.on('drag', (event, d) => {
					d.fx = event.x
					d.fy = event.y
				})
				.on('end', (event, d) => {
					if (!event.active) simulation.alphaTarget(0)
					d.fx = null
					d.fy = null
				})
			)

		// Smooth continuous floating animation using sine waves
		let animationFrame
		let time = 0

		const animate = () => {
			time += 0.02  // Controls animation speed

			circles
				.attr('cx', d => {
					// Each node has unique phase based on its id
					const phase = d.id * 0.5
					const floatX = Math.sin(time + phase) * 3  // Horizontal drift
					return d.x + floatX
				})
				.attr('cy', d => {
					const phase = d.id * 0.5
					const floatY = Math.cos(time * 0.7 + phase) * 4  // Vertical drift (slower)
					return d.y + floatY
				})

			animationFrame = requestAnimationFrame(animate)
		}

		simulation.on('tick', () => {
			circles.attr('cx', d => d.x).attr('cy', d => d.y)
		})

		// Start smooth animation after simulation settles
		setTimeout(() => {
			simulation.stop()
			animate()
		}, 3000)

		return () => {
			cancelAnimationFrame(animationFrame)
			simulation.stop()
		}
	}, [data, hasData])

	return (
		<div className="map-container">
			<div className="map-header">
				<h2 className="map-title">
					⚔️ Tug of War
					<span className="map-badge">POLARITY MAP</span>
				</h2>
			</div>

			<div style={ { position: 'relative' } }>
				<div className="map-visualization" style={ { background: '#0a0a0a' } }>
					{ !hasData ? (
						<div className="map-placeholder">
							<div style={ { fontSize: '3rem', marginBottom: '1rem' } }>🌌</div>
							<div>No data yet.</div>
							<div>Search for a topic to see the polarity map.</div>
						</div>
					) : (
						<svg
							ref={ svgRef }
							width={ 780 }
							height={ 600 }
							style={ { display: 'block' } }
						>
							{/* D3 will render nodes here */ }
						</svg>
					) }
				</div>

				{/* Tooltip showing full tweet on hover */ }
				{ tooltipData && (
					<div style={ {
						position: 'absolute',
						top: 10,
						right: 10,
						width: 280,
						background: '#fff8e7',
						border: '3px solid #1a1a1a',
						boxShadow: '4px 4px 0px #1a1a1a',
						padding: '1rem',
						zIndex: 10,
						fontFamily: "'Space Mono', monospace"
					} }>
						<div style={ {
							fontSize: '0.75rem',
							fontWeight: 'bold',
							color: emotionColors[tooltipData?.emotion] || '#6b7280',
							marginBottom: '0.5rem'
						} }>
							{ tooltipData?.handle?.split('\n')[0] }
							<span style={ {
								marginLeft: '0.5rem',
								background: emotionColors[tooltipData?.emotion] || '#6b7280',
								color: '#fff',
								padding: '2px 6px',
								fontSize: '0.625rem',
								borderRadius: '3px'
							} }>
								{ emotionEmojis[tooltipData?.emotion] } { tooltipData?.emotion }
							</span>
						</div>
						<div style={ { fontSize: '0.7rem', marginBottom: '0.5rem', display: 'flex', gap: '0.5rem' } }>
							<span style={ {
								padding: '2px 6px',
								background: tooltipData?.sentiment_score > 0 ? '#22c55e' : tooltipData?.sentiment_score < 0 ? '#ef4444' : '#6b7280',
								color: '#fff',
								borderRadius: '3px'
							} }>
								{ tooltipData?.sentiment } ({ tooltipData?.sentiment_score?.toFixed(2) })
							</span>
						</div>
						<div style={ { fontSize: '0.8rem', lineHeight: 1.4 } }>
							{ tooltipData?.text?.slice(0, 200) }
							{ tooltipData?.text?.length > 200 ? '...' : '' }
						</div>
					</div>
				) }
			</div>
		</div>
	)
}

export default ClusterMap
