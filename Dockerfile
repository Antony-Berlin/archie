# NeuralArch-Bench — OpenEnv environment
# Base: official PyTorch image (CPU-only for HF Spaces free tier)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# HF Spaces runs as a non-root user; create one to match
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── system deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy project (includes data/ CSVs) ────────────────────────────────────────
COPY . .

# Verify tabular datasets are present and loadable
RUN python - <<'EOF'
import pandas as pd
from pathlib import Path
for name in ["iris", "wine", "breast_cancer"]:
    p = Path(f"/app/data/{name}.csv")
    df = pd.read_csv(p)
    print(f"{name}.csv: {len(df)} rows, {len(df.columns)} cols OK")
EOF

# Fix ownership so appuser can write temp dirs
RUN chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
