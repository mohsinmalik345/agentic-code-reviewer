FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app

COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade --retries 10 --timeout 120 pip \
    && python -c "import pathlib, subprocess, sys, tomllib; project = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--retries', '10', '--timeout', '120', *project['dependencies']])"

COPY src ./src
RUN python -m pip install --no-deps .

COPY config ./config
COPY migrations ./migrations
RUN mkdir -p /app/artifacts /app/repositories /app/internal-artifacts /app/internal-repositories \
    && chown -R app:app /app

USER app
EXPOSE 4000 4100
CMD ["code-intel", "serve", "--config", "/app/config/container.yaml"]

