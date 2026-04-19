FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install runtime dependencies directly (avoids flat-layout package discovery issues)
RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    numpy>=1.26 \
    httpx>=0.27 \
    lightgbm>=4.3 \
    xgboost>=2.0 \
    pandas>=2.2 \
    torch>=2.3 \
    python-dotenv>=1.0

COPY api/      api/
COPY models/   models/
COPY backtest/ backtest/
COPY features/ features/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
