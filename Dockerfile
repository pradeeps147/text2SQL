# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ARG APP_VERSION=dev
LABEL org.opencontainers.image.title="Tenarai logistics intelligence" \
      org.opencontainers.image.description="Retrieval-grounded logistics analytics API and Streamlit UI" \
      org.opencontainers.image.version="${APP_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    SERVICE_ROLE=backend

WORKDIR /app

RUN groupadd --gid 10001 tenarai \
    && useradd --uid 10001 --gid tenarai --create-home --shell /usr/sbin/nologin tenarai

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=tenarai:tenarai . .
RUN mkdir -p data/processed data/uploads logs \
    && chown -R tenarai:tenarai data logs

USER tenarai

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('PORT', '8000'); path='/_stcore/health' if os.environ.get('SERVICE_ROLE') == 'ui' else '/api/v1/health'; urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=4)" || exit 1

CMD ["/bin/sh", "-c", "exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
