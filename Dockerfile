FROM ubuntu:focal-20210827

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    curl \
    python3-pip

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
ENV PATH="$HOME/.poetry/bin:$PATH"

COPY poetry.lock pyproject.toml /root/
COPY pyproject.toml /root/

WORKDIR /root

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction
