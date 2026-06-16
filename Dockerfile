# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /opt/orchid

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .[agentic,viz,observability] \
    && useradd -ms /bin/bash orchid \
    && chown -R orchid:orchid /opt/orchid

USER orchid

EXPOSE 8000 8081 9090

# serve defaults to binding 127.0.0.1 (safe by default); inside a container we
# must bind all interfaces so the published ports are reachable.
ENTRYPOINT ["python", "-m", "orchid_ranker.cli.serve", "--host", "0.0.0.0"]
