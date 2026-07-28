.PHONY: install data test

install:
	pip install -r requirements.txt

data:
	python -m src.data.pipeline

test:
	pytest tests/ -v
