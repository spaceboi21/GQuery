# XIA — Excel Intelligence Agent
# Single-stage image. Frontend is static HTML, so no Node build step needed.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XIA_APP_HOST=0.0.0.0 \
    XIA_APP_PORT=8000 \
    XIA_DATA_PATH=/data \
    XIA_OPEN_BROWSER=0

WORKDIR /app

# curl is only used for the healthcheck below; ~1MB.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first so layer is cached when only source changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY main.py ./

# Mount the user's Excel/CSV folder here. xia_knowledge.db lives inside it,
# so re-builds don't lose the index.
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/stats || exit 1

CMD ["python", "main.py"]
