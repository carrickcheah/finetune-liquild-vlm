evaluate:
	uv run modal run -m ft_vlm.cli.evaluate --config-file-name $(config)

fine-tune:
	uv run modal run -m ft_vlm.cli.train --config-file-name $(config)

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -v -x

test-cov:
	uv run pytest tests/ -v --cov=src/ft_vlm --cov-report=term-missing
