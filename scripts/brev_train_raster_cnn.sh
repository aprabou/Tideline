#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/brev_train_raster_cnn.sh [epochs] [batch_size] [lr] [patience]
# Example:
#   bash scripts/brev_train_raster_cnn.sh 50 8 1e-3 8

EPOCHS="${1:-50}"
BATCH_SIZE="${2:-8}"
LR="${3:-1e-3}"
PATIENCE="${4:-8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[error] .venv/bin/python not found. Create venv and install dependencies first."
  echo "        python -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

if [[ -z "${GCS_BUCKET:-}" ]]; then
  echo "[error] GCS_BUCKET is required for raster_cnn training."
  echo "        export GCS_BUCKET=<your-bucket-name>"
  exit 1
fi

echo "[info] Python: $ROOT_DIR/.venv/bin/python"
echo "[info] GCS_BUCKET=${GCS_BUCKET}"

if [[ -n "${DATABRICKS_HOST:-}" && -n "${DATABRICKS_TOKEN:-}" ]]; then
  echo "[info] Databricks MLflow tracking detected via DATABRICKS_HOST + DATABRICKS_TOKEN"
elif [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "[info] MLflow tracking URI detected: ${MLFLOW_TRACKING_URI}"
else
  echo "[warn] No MLflow env vars found. Training still runs; tracking defaults may be local."
fi

echo "[info] Checking CUDA availability..."
"$ROOT_DIR/.venv/bin/python" - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

echo "[info] Starting raster CNN training..."
"$ROOT_DIR/.venv/bin/python" -m models.raster_cnn train \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --patience "$PATIENCE"

echo "[info] Training done. Running 2023 inference..."
"$ROOT_DIR/.venv/bin/python" -m models.raster_cnn infer --batch-size "$BATCH_SIZE"

echo "[info] Complete."
