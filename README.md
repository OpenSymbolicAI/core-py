# OpenSymbolicAI Core

OpenSymbolicAI Core Runtime

## Development

### Setup

```bash
uv sync
```

### Pre-commit Hooks

```bash
uv run pre-commit install          # Install hooks (one-time setup)
uv run pre-commit run --all-files  # Run all hooks manually
```

Hooks run automatically on `git commit` and include:
- **ruff** - linting with auto-fix
- **ruff-format** - code formatting
- **mypy** - type checking
- **pytest** - run tests
- Trailing whitespace, EOF fixes, YAML/TOML validation

### Commands

```bash
uv run ruff check .        # Lint
uv run ruff check --fix .  # Lint and auto-fix
uv run mypy src            # Type check
uv run pytest              # Run tests
```
