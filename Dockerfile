FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY serve_model_API.py .
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000
ENV MODEL_NAME=LicensePlate-Detector
ENV MODEL_CHECK_INTERVAL=300
EXPOSE 3000
CMD ["uvicorn", "serve_model_API:app", "--host", "0.0.0.0", "--port", "3000"]