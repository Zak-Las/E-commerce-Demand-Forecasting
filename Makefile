# Convenience commands

.PHONY: download-m5 api test lint format train-nbeats

ENV?=Zak-Las

_download_check:
	@command -v kaggle >/dev/null 2>&1 || { echo 'kaggle CLI not found. Install inside env: pip install kaggle'; exit 1; }

create-env:
	conda env create -f environment.yml || echo "Env may already exist"

update-env:
	conda env update -f environment.yml --prune

activate:
	@echo "Run: conda activate $(ENV)"

download-m5: _download_check
	python -m src.data.download_m5 --output data/raw/m5

api:
	uvicorn src.service.app:app --reload

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests

train-nbeats:
	python scripts/train_nbeats.py
