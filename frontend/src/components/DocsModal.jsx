import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import frontendDocs from '../../../docs/FRONTEND_DOCS.md?raw'
import backendDocs from '../../../docs/BACKEND_API.md?raw'
import './DocsModal.css'

const DocsModal = ({ onClose }) => {
	const [activeTab, setActiveTab] = useState('frontend')

	return (
		<div className="docs-modal-overlay" onClick={ onClose }>
			<div className="docs-modal-content" onClick={ e => e.stopPropagation() }>
				<div className="docs-modal-header">
					<div className="docs-tabs">
						<button
							className={ `docs-tab ${activeTab === 'frontend' ? 'active' : ''}` }
							onClick={ () => setActiveTab('frontend') }
						>
							Frontend Docs
						</button>
						<button
							className={ `docs-tab ${activeTab === 'backend' ? 'active' : ''}` }
							onClick={ () => setActiveTab('backend') }
						>
							Backend API
						</button>
					</div>
					<button className="docs-close-btn" onClick={ onClose }>&times;</button>
				</div>

				<div className="docs-body">
					<div className="markdown-body">
						<ReactMarkdown>
							{ activeTab === 'frontend' ? frontendDocs : backendDocs }
						</ReactMarkdown>
					</div>
				</div>
			</div>
		</div>
	)
}

export default DocsModal
