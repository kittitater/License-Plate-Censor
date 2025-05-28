from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import numpy as np
from io import BytesIO
import os
import logging
import requests

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "LicensePlate-Detector")
CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", 300))  # 5 minutes

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for model and version
current_model = None
current_version = None

def load_latest_model():
    global current_model, current_version
    client = mlflow.tracking.MlflowClient()
    try:
        # Search for all versions of the model and sort by version number
        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not model_versions:
            raise Exception(f"No versions found for {MODEL_NAME}")
        
        # Get the latest version (highest version number)
        latest_version = max([int(mv.version) for mv in model_versions])
        if current_version != str(latest_version):
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            logger.info(f"Loading model from {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)
            current_model = YOLO(model)  # Wrap for YOLO
            current_version = str(latest_version)
            logger.info(f"Loaded model version {latest_version} for {MODEL_NAME}")
        else:
            logger.info(f"Model version {latest_version} already loaded")
    except Exception as e:
        logger.error(f"Failed to load latest model: {str(e)}")
        raise

# Health check function
def check_mlflow_connectivity():
    try:
        response = requests.get(MLFLOW_TRACKING_URI, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Load model at startup
load_latest_model()

# Background task to check for the latest model version
scheduler = BackgroundScheduler()
scheduler.add_job(load_latest_model, "interval", seconds=CHECK_INTERVAL)
scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "mlflow_server": "accessible",
        "model": "loaded",
        "model_name": MODEL_NAME,
        "model_version": current_version
    }
    
    # Check MLflow server connectivity
    if not check_mlflow_connectivity():
        health_status["status"] = "unhealthy"
        health_status["mlflow_server"] = "inaccessible"
    
    # Check model availability
    if current_model is None:
        health_status["status"] = "unhealthy"
        health_status["model"] = "not loaded"
        health_status["model_version"] = None
    
    return JSONResponse(
        content=health_status,
        status_code=200 if health_status["status"] == "healthy" else 503
    )

@app.post("/blur_license_plate")
async def blur_license_plate(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        results = current_model(img)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]
            roi = cv2.GaussianBlur(roi, (23, 23), 30)
            img[y1:y2, x1:x2] = roi
        
        _, img_encoded = cv2.imencode('.jpg', img)
        return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))