from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import mlflow
import mlflow.pytorch
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import numpy as np
from io import BytesIO
import os
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "YOLO_LicenseCensor")
CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", 300))  # 5 minutes

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for model and version
current_model = None
current_version = None

def load_production_model():
    global current_model, current_version
    client = mlflow.tracking.MlflowClient()
    try:
        production_model = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not production_model:
            raise Exception(f"No model found in 'Production' stage for {MODEL_NAME}")
        
        latest_version = production_model[0].version
        if current_version != latest_version:
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            logger.info(f"Loading model from {model_uri}")
            model = mlflow.pytorch.load_model(model_uri)
            current_model = YOLO(model)  # Wrap for YOLO
            current_version = latest_version
            logger.info(f"Loaded model version {latest_version} for {MODEL_NAME}")
        else:
            logger.info(f"Model version {latest_version} already loaded")
    except Exception as e:
        logger.error(f"Failed to load Production model: {str(e)}")
        raise

# Load model at startup
load_production_model()

# Background task to check for new Production model
scheduler = BackgroundScheduler()
scheduler.add_job(load_production_model, "interval", seconds=CHECK_INTERVAL)
scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()

@app.post("/license-censor")
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