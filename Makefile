TESTS = tests

test:
	pytest $(TESTS)

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

