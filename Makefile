# Convenience commands
.PHONY: install notebook

install:
	python -m pip install -U pip
	pip install -r requirements.txt

notebook:
	jupyter notebook
