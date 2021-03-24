FROM python:3.8

WORKDIR /app

ENV FLASK_ENV=development
ENV FLASK_APP=catanatron_server/server.py
ENV FLASK_RUN_HOST=0.0.0.0

# We copy just the dependencies first to leverage Docker cache
COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN pip install .

EXPOSE 5000

CMD flask run
