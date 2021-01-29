FROM python:3.8

RUN pip install pipenv

WORKDIR /app

COPY Pipfile /app
COPY Pipfile.lock /app
RUN pipenv lock --requirements > requirements.txt
RUN pip install -r /app/requirements.txt

ENV FLASK_ENV=development
ENV FLASK_APP=catanatron_server/server.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY . .

RUN pip install .

EXPOSE 5000

CMD flask run
