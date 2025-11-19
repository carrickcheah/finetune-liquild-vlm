evaluate:
	uv run modal run -m ft_vlm.cli.evaluate --config-file-name $(config)

fine-tune:
	uv run modal run -m ft_vlm.cli.train --config-file-name $(config)
