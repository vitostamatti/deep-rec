install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-dev:
	pip install -r requirements.dev.txt


format:
	black deeprec/*.py && black tests/*.py	


lint:
	pylint --disable=R,C deeprec/*.py


test:
	python -m pytest -vv --cov=tests tests

