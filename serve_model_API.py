from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://ec2-13-215-137-215.ap-southeast-1.compute.amazonaws.com:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "LicensePlate-Detector")
CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", 300))  # 5 minutes

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for model and version
current_model = None
current_version = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kittitater.github.io","http://127.0.0.1:5500"],  # Allows all origins (replace with specific origins for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

def load_latest_model():
    global current_model, current_version
    client = mlflow.tracking.MlflowClient()
    try:
        # Search for all versions of the model
        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not model_versions:
            raise Exception(f"No versions found for {MODEL_NAME}")
        
        # Get the latest version (highest version number)
        latest_version = max([int(mv.version) for mv in model_versions])
        if current_version != str(latest_version):
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            logger.info(f"Loading model from {model_uri}")
            
            # Download model artifacts
            model_path = mlflow.artifacts.download_artifacts(model_uri)
            logger.info(f"Artifact directory: {model_path}")
            logger.info(f"Directory contents: {os.listdir(model_path)}")
            
            # Search for best.pt recursively
            pt_file_path = None
            for root, _, files in os.walk(model_path):
                if "best.pt" in files:
                    pt_file_path = os.path.join(root, "best.pt")
                    break
            
            # Verify the .pt file exists
            if not pt_file_path or not os.path.exists(pt_file_path):
                # Log all files for debugging
                all_files = []
                for root, _, files in os.walk(model_path):
                    for file in files:
                        all_files.append(os.path.join(root, file))
                logger.error(f"All files in artifacts: {all_files}")
                raise Exception(f"No best.pt file found in artifacts for {MODEL_NAME} version {latest_version}")
            
            logger.info(f"Loading YOLO model from {pt_file_path}")
            # Load the YOLO model from the .pt file
            current_model = YOLO(pt_file_path)
            current_version = str(latest_version)
            logger.info(f"Loaded model version {latest_version} for {MODEL_NAME}")
        else:
            logger.info(f"Model version {latest_version} already loaded")
    except Exception as e:
        logger.error(f"Failed to load latest model: {str(e)}")
        # Don't raise, allow app to continue with previous model (if any)

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
        if current_model is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        img_data = await file.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = current_model(img)

        h, w = img.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

            roi = img[y1:y2, x1:x2]
            roi = cv2.GaussianBlur(roi, (51, 51), 60)
            img[y1:y2, x1:x2] = roi

            # Draw green rectangle border
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, img_encoded = cv2.imencode('.jpg', img)
        return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))