# Contributing

Thanks for your interest in contributing to Blackwell Inference Server.

## Development setup

```bash
git clone https://github.com/Leslie360/blackwell-inference-server.git
cd -- blackwell-inference-server
pip install -e .[dev]
```

## Running tests

```bash
pytest tests/
```

## Code style

- Format with `black` and `isort`
- Lint with `ruff`
- Type-check with `mypy`

## Pull requests

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run tests and lint
5. Submit a PR with a clear description

## Reporting bugs

Please use the GitHub issue templates and include:

- GPU model and CUDA version
- PyTorch version
- Command to reproduce
- Full error log
