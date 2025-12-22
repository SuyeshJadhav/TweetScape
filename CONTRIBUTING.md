# Contributing to TweetScape

First off, thank you for considering contributing to TweetScape! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Find an Issue

- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it
- If you want to work on something not listed, open an issue first to discuss

### Types of Contributions

| Type | Description |
|------|-------------|
| 🐛 **Bug Reports** | Found a bug? Open an issue with reproduction steps |
| ✨ **Feature Requests** | Have an idea? Discuss it in an issue first |
| 📖 **Documentation** | Improve docs, fix typos, add examples |
| 🧪 **Tests** | Add test coverage, fix flaky tests |
| 💻 **Code** | Fix bugs, implement features |

## Development Setup

### Prerequisites

```bash
# Required
Python 3.10+
Node.js 18+
Git

# Optional (for AI features)
Ollama with gemma3:4b model
```

### Local Setup

```bash
# 1. Fork and clone
git clone https://github.com/SuyeshJadhav/tweetscape.git
cd tweetscape

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# 3. Frontend setup
cd ../frontend
npm install

# 4. Run both (use two terminals)
# Terminal 1 - Backend
cd backend && uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend && npm run dev
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Screenshots** if applicable
- **Environment** (OS, Python version, Node version)

### Suggesting Features

Feature requests are welcome! Please provide:

- **Clear use case** - What problem does it solve?
- **Proposed solution** - How should it work?
- **Alternatives** - Have you considered other approaches?

## Pull Request Process

### 1. Branch Naming

```bash
feature/add-export-pdf
bugfix/fix-tooltip-overflow
docs/update-readme
refactor/simplify-pipeline
```

### 2. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add PDF export for visualizations
fix: resolve tooltip overflow on small screens
docs: update installation instructions
refactor: simplify pipeline orchestration
test: add coverage for emotion detection
```

### 3. Before Submitting

- [ ] Code follows the style guidelines
- [ ] Self-reviewed the code
- [ ] Added comments for complex logic
- [ ] Updated documentation if needed
- [ ] No new warnings introduced
- [ ] Tests pass locally (if applicable)

### 4. PR Description

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Has This Been Tested?
Describe testing approach

## Screenshots (if applicable)
```

## Style Guidelines

### Python (Backend)

```python
# Use type hints
def process_topic_data(topic: str) -> dict:
    """
    Docstring with description.
    
    Args:
        topic: Search topic string
    
    Returns:
        Result dict with all data
    """
    pass

# Use descriptive variable names
sentiment_score = get_sentiment(text)  # Good
s = get_sentiment(text)                 # Bad

# Use logging, not print
logger.info(f"Processing {len(tweets)} tweets")
```

### JavaScript/React (Frontend)

```jsx
// Functional components with hooks
function ClusterMap({ data, onPointClick }) {
    const [tooltipData, setTooltipData] = useState(null)
    // ...
}

// Destructure props
// Good
function Component({ title, items }) {}

// Avoid
function Component(props) {
    const title = props.title
}
```

### CSS

- Use CSS variables for theming
- Follow BEM-like naming: `.map-container`, `.map-title`, `.map-badge`
- Mobile-first responsive design

## Questions?

Feel free to open an issue or reach out if you have questions. We're happy to help!

---

Thank you for contributing! 🙏
