FROM python:3.11-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by `implicit` (ALS), not present in the slim base.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY data/raw/spotify_tracks/ data/raw/spotify_tracks/

# als_model.pkl / joined_train.parquet aren't copied in (too large for a lean image,
# and gitignored anyway) — src/artifacts.py downloads them from Hugging Face Hub on
# first request instead. See README's Architecture section.

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
