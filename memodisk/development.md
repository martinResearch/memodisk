
**Development version**

```bash
pip install git+https://github.com/martinResearch/memodisk.git
```

## Development Setup

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management. Install it first:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/martinResearch/memodisk.git
cd memodisk

# Create virtual environment with uv
uv sync --all-extras

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

### Development Workflow

#### Code Quality Checks

We use [Ruff](https://github.com/astral-sh/ruff) for fast linting and formatting:

```bash
# Check code quality (linting)
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Check formatting without changing files
ruff format --check .
```

#### Type Checking

```bash
# Run type checking with mypy
mypy memodisk/
```

#### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=memodisk --cov-report=html

# Run specific test file
pytest tests/test_specific.py -v
```

#### Complete Development Workflow

Run all checks in sequence:

```bash
# 1. Format code
ruff format .

# 2. Fix linting issues
ruff check --fix .

# 3. Check for remaining issues
ruff check .

# 4. Run type checking
mypy memodisk/

# 5. Run tests
pytest tests/ -v --cov=memodisk
```
