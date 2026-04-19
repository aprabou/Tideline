# Tideline API — Cloud Run container
# Build:  docker build -t tideline-api .
# Run:    docker run -p 8080:8080 -e GEMINI_API_KEY=... tideline-api

FROM python:3.12-slim

WORKDIR /app

# System deps for numpy / pandas native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install only the runtime dependencies (not dev extras)
RUN pip install --no-cache-dir ".[api]"

# Copy source
COPY api/      api/
COPY models/   models/
COPY backtest/ backtest/

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
