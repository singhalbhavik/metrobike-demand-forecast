.PHONY: install data train test

install:
	pip install -r requirements.txt

data:
	python -m src.data.pipeline

train:
	python -m src.models.pipeline

test:
	pytest tests/ -v
