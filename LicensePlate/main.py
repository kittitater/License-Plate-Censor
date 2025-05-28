from ultralytics import YOLO
import torch
import mlflow
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import pandas as pd
import gc
import time
import psutil
import GPUtil

def log_system_metrics():
    """Logs CPU, RAM, and GPU utilization metrics to MLflow."""
    cpu_percent = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100
        gpu_mem = gpu.memoryUtil * 100
        mlflow.log_metric("gpu_load_percent", gpu_load)
        mlflow.log_metric("gpu_memory_percent", gpu_mem)
    else:
        mlflow.log_metric("gpu_load_percent", 0)
        mlflow.log_metric("gpu_memory_percent", 0)

    mlflow.log_metric("cpu_percent", cpu_percent)
    mlflow.log_metric("ram_percent", ram_usage)

def main():
    torch.cuda.empty_cache()
    gc.collect()

    remote_server_uri = "http://ec2-13-215-137-215.ap-southeast-1.compute.amazonaws.com:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("LicensePlate-Detection")

    run_name = "lp-detector-"
    data_yaml_path = "synthetic_data/data.yaml"
    output_dir = os.path.abspath(f"runs/detect/{run_name}")

    params = {
        "model": "yolov8n.pt",
        "data": data_yaml_path,
        "epochs": 10,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "name": run_name,
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "project": "License Plate Detection",
            "framework": "YOLOv8",
            "phase": "training"
        })
        mlflow.log_params(params)

        # Train model
        model = YOLO(params["model"])
        results = model.train(
            data=params["data"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            workers=params["workers"],
            name=params["name"],
            plots=False
        )

        # Log metrics from results object
        if hasattr(results, 'results_dict'):  # Updated to match Ultralytics API
            metrics = results.results_dict
            mlflow.log_metrics({
                "metrics/precision": metrics.get("metrics/precision(B)", 0),
                "metrics/recall": metrics.get("metrics/recall(B)", 0),
                "metrics/mAP50": metrics.get("metrics/mAP50(B)", 0),
                "metrics/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
            })
        else:
            print("No metrics available in results object.")

        # Log artifacts
        weights_dir = os.path.join(output_dir, "weights")
        best_model = os.path.join(weights_dir, "best.pt")
        last_model = os.path.join(weights_dir, "last.pt")
        results_csv = os.path.join(output_dir, "results.csv")

        # Log the best model with PyTorch flavor
        if os.path.exists(best_model):
            registered_model_name = "LicensePlate-Detector"
            mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path="yolo_model",
                registered_model_name=registered_model_name
            )
            print(f"Logged best.pt to MLflow with PyTorch flavor under {registered_model_name}")

        # Optionally log the last model as an artifact
        if os.path.exists(last_model):
            mlflow.log_artifact(last_model, artifact_path="models")

        # Log training results CSV
        if os.path.exists(results_csv):
            try:
                df = pd.read_csv(results_csv)
                last_row = df.iloc[-1].to_dict()
                numeric_metrics = {}
                for k, v in last_row.items():
                    try:
                        numeric_metrics[k] = float(v)
                    except ValueError:
                        continue
                mlflow.log_metrics(numeric_metrics)
                mlflow.log_artifact(results_csv, artifact_path="metrics")
            except Exception as e:
                print(f"Error reading or logging results.csv: {e}")
        else:
            print("results.csv not found.")
        
        log_system_metrics()
        time.sleep(15)

if __name__ == "__main__":
    main()