FROM python:3.12-slim

WORKDIR /app

# Install dependencies via UV
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/cache data/embeddings data/vectordb logs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
