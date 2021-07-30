clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	# find . -name '*~' -exec rm -f {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

build-catanatron: clean
	python catanatron_core/setup.py sdist bdist_wheel
	ls -l dist
	twine check dist/*


build-catanatron-gym: clean
	python catanatron_gym/setup.py sdist bdist_wheel
	ls -l dist
	twine check dist/*

upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-production:
	twine upload dist/*