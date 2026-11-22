# 1. Build the Docker image
docker build -t onnx-digit-api:latest .

# 2. Run the container locally
docker run -d -p 8000:8000 --name digit-api onnx-digit-api:latest

# 3. Verify health endpoint
curl http://localhost:8000/health
