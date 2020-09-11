# catanatron

Catan Player

## Usage

On one tab:

```
pipenv install
pipenv shell
export FLASK_ENV=development
export FLASK_APP=server.py
flask run
```

On another tab:

```
cd ui/
yarn install
yarn start
```

## To run tests

```
coverage run --source=catanatron -m pytest tests/ && coverage report
```
