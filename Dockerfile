# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /opt/orchid

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .[agentic,viz,observability] \
    && useradd -ms /bin/bash orchid \
    && chown -R orchid:orchid /opt/orchid

USER orchid

EXPOSE 8000

ENTRYPOINT ["python", "-m", "orchid_ranker.cli.evaluate", "--help"]
