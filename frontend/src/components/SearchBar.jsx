import { useState } from 'react'

function SearchBar({ onSearch, loading }) {
	const [topic, setTopic] = useState('')
	const [limit, setLimit] = useState(10)

	const handleSubmit = (e) => {
		e.preventDefault()
		console.log('SearchBar: Calling onSearch with topic:', topic, 'and limit:', limit)
		onSearch(topic, limit)
	}

	return (
		<div className="search-container">
			<h1 className="search-title">
				Analyze <span className="search-title-accent">Narratives</span>
			</h1>

			<form className="search-form" onSubmit={ handleSubmit }>
				<div className="input-group">
					<label className="input-label">Topic / Keyword</label>
					<input
						type="text"
						className="input-field"
						placeholder="Enter topic (e.g., ChatGPT, Bitcoin...)"
						value={ topic }
						onChange={ (e) => setTopic(e.target.value) }
						disabled={ loading }
					/>
				</div>

				<div className="input-group">
					<label className="input-label">Limit</label>
					<input
						type="number"
						className="input-field input-number"
						placeholder="10"
						min="5"
						max="100"
						value={ limit }
						onChange={ (e) => setLimit(Number(e.target.value)) }
						disabled={ loading }
					/>
				</div>

				<button
					type="submit"
					className="btn-primary"
					disabled={ loading }
					onClick={ handleSubmit }
				>
					{ loading ? '⏳ SCANNING...' : '🔍 ANALYZE' }
				</button>
			</form>
		</div>
	)
}

export default SearchBar
