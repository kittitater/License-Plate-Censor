FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve_model_API.py .
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MODEL_NAME=YOLOv8-LicensePlate-Detection
ENV MODEL_CHECK_INTERVAL=300
EXPOSE 80
CMD ["uvicorn", "serve_model_API:app", "--host", "0.0.0.0", "--port", "80"]
