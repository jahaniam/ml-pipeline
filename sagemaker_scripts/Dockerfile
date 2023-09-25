FROM python:3.9-slim

WORKDIR /app

# install dependencies
COPY Pipfile $DAGSTER_APP
COPY Pipfile.lock $DAGSTER_APP
RUN pip install --upgrade pip
RUN pip install pipenv
COPY . .
RUN pipenv install --deploy --system
