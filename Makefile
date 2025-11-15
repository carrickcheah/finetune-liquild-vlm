evaluate:
	uv run modal run -m ft_vlm.cli.evaluate --config-file-name $(config)

report:
	uv run jupyter notebook notebooks/visualize_evals.ipynb

fine-tune:
	uv run modal run -m ft_vlm.cli.train --config-file-name $(config)

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

code-fixes: lint format
