FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr or writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORT_THREADS=2

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY digit_cnn.onnx app.py ./

EXPOSE 8000

# Run with Uvicorn (keep workers=1 or 2 when ONNX Runtime uses multi-threading)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
