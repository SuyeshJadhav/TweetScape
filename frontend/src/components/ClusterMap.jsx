import * as d3 from 'd3'
import { useRef, useState, useEffect } from 'react'

function ClusterMap({ data, onPointClick }) {
	const svgRef = useRef()
	const [tooltipData, setTooltipData] = useState(null)
	const hasData = data && data.length > 0

	// Color palette for sentiment
	const clusterColors = {
		0: '#ef4444', // Negative - red
		1: '#6b7280', // Neutral - gray
		2: '#22c55e', // Positive - green
	}

	const clusterNames = {
		0: '😠 Negative',
		1: '😐 Neutral',
		2: '😊 Positive'
	}

	useEffect(() => {
		if (!hasData || !svgRef.current) return

		const width = 780
		const height = 600

		// Adaptive node size based on count
		const nodeCount = data.length
		const nodeRadius = Math.max(6, Math.min(16, 200 / Math.sqrt(nodeCount)))

		// Circular cluster centers
		const centerX = width / 2
		const centerY = height / 2
		const clusterRadius = 180 // Distance from center to cluster center

		// Positions in a triangle: Negative top-left, Neutral top-right, Positive bottom
		const clusterCenters = {
			0: { x: centerX - clusterRadius, y: centerY - clusterRadius * 0.5 }, // Negative - top left
			1: { x: centerX + clusterRadius, y: centerY - clusterRadius * 0.5 }, // Neutral - top right
			2: { x: centerX, y: centerY + clusterRadius * 0.8 } // Positive - bottom
		}

		const nodes = data.map((tweet, i) => {
			const target = clusterCenters[tweet.cluster] || { x: centerX, y: centerY }
			return {
				id: i,
				...tweet,
				// Start nodes near their target cluster with randomness
				x: target.x + (Math.random() - 0.5) * 100,
				y: target.y + (Math.random() - 0.5) * 100
			}
		})

		const simulation = d3.forceSimulation(nodes)
			.force('charge', d3.forceManyBody().strength(-15))
			.force('collide', d3.forceCollide(nodeRadius + 2))
			.force('x', d3.forceX(d => clusterCenters[d.cluster]?.x || centerX).strength(0.12))
			.force('y', d3.forceY(d => clusterCenters[d.cluster]?.y || centerY).strength(0.12))
			.alphaDecay(0.02)
			.velocityDecay(0.25)

		const svg = d3.select(svgRef.current)
		svg.selectAll('*').remove()

		// Zone labels - positioned near cluster centers
		const zones = [
			{ x: clusterCenters[0].x, y: clusterCenters[0].y - 80, label: '😠 NEGATIVE', color: '#ef4444' },
			{ x: clusterCenters[1].x, y: clusterCenters[1].y - 80, label: '😐 NEUTRAL', color: '#6b7280' },
			{ x: clusterCenters[2].x, y: clusterCenters[2].y + 100, label: '😊 POSITIVE', color: '#22c55e' }
		]

		svg.selectAll('.zone-label')
			.data(zones)
			.enter()
			.append('text')
			.attr('class', 'zone-label')
			.attr('x', d => d.x)
			.attr('y', d => d.y)
			.attr('text-anchor', 'middle')
			.attr('fill', d => d.color)
			.attr('font-size', '14px')
			.attr('font-family', 'Space Mono, monospace')
			.attr('font-weight', 'bold')
			.text(d => d.label)

		// Glow filters for extreme sentiments
		const defs = svg.append('defs')

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
				const intensity = Math.abs(d.sentiment_score || 0)
				const opacity = 0.4 + intensity * 0.6
				if (d.cluster === 2) return `rgba(34, 197, 94, ${opacity})`
				if (d.cluster === 0) return `rgba(239, 68, 68, ${opacity})`
				return `rgba(107, 114, 128, 0.6)`
			})
			.attr('stroke', '#1a1a1a')
			.attr('stroke-width', 2)
			.attr('filter', d => {
				const intensity = Math.abs(d.sentiment_score || 0)
				if (intensity > 0.7) {
					return d.sentiment_score > 0 ? 'url(#glow-green)' : 'url(#glow-red)'
				}
				return null
			})
			.style('cursor', 'pointer')
			.on('mouseover', function (event, d) {
				d3.select(this).attr('r', nodeRadius * 1.4).attr('stroke-width', 3)
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
		// Gentle random floating motion - slower with more amplitude
		const float = () => {
			nodes.forEach(node => {
				node.vx += (Math.random() - 0.5) * 0.6
				node.vy += (Math.random() - 0.5) * 0.6
			})
			simulation.alpha(0.03).restart()
		}
		const floatInterval = setInterval(float, 3000)

		simulation.on('tick', () => {
			circles.attr('cx', d => d.x).attr('cy', d => d.y)
		})

		return () => {
			clearInterval(floatInterval)
			simulation.stop()
		}
	}, [data, hasData])

	return (
		<div className="map-container">
			<div className="map-header">
				<h2 className="map-title">
					🗺️ Semantic Landscape
					<span className="map-badge">INTERACTIVE</span>
				</h2>
			</div>

			<div style={ { position: 'relative' } }>
				<div className="map-visualization" style={ { background: '#0a0a0a' } }>
					{ !hasData ? (
						<div className="map-placeholder">
							<div style={ { fontSize: '3rem', marginBottom: '1rem' } }>🌌</div>
							<div>No data yet.</div>
							<div>Search for a topic to see the narrative map.</div>
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
							color: clusterColors[tooltipData?.cluster],
							marginBottom: '0.5rem'
						} }>
							{ tooltipData?.handle?.split('\n')[0] }
							<span style={ {
								marginLeft: '0.5rem',
								background: clusterColors[tooltipData?.cluster],
								color: '#fff',
								padding: '2px 6px',
								fontSize: '0.625rem'
							} }>
								{ clusterNames[tooltipData?.cluster] }
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
