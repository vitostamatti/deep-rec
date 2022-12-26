install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-dev:
	pip install -r requirements.dev.txt


format:
	black src/*.py && black tests/*.py	


lint:
	pylint --disable=R,C src/*.py


test:
	python -m pytest -vv --cov=tests tests


all: install format lint test