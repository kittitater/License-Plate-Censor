import os
import time
import torch
import mlflow
import gc
import psutil
import GPUtil
import yaml
import argparse
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet18
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import utils
from engine import train_one_epoch, evaluate

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

def log_system_metrics():
    try:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        if gpus:
            mlflow.log_metric("gpu_load_percent", gpus[0].load * 100)
            mlflow.log_metric("gpu_memory_percent", gpus[0].memoryUtil * 100)
        mlflow.log_metric("cpu_percent", cpu)
        mlflow.log_metric("ram_percent", ram)
    except Exception as e:
        print(f"Warning: Failed to log system metrics: {e}")

def log_artifact_with_retry(local_path, artifact_path="models", max_retries=3, retry_delay=10):
    """
    Log artifact with retry mechanism and fallback options
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting to upload model (attempt {attempt + 1}/{max_retries})...")
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            print("Model uploaded successfully!")
            return True
        except Exception as e:
            print(f"Upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All upload attempts failed. Model saved locally only.")
                return False
    return False

def load_data_yaml(path="data.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: '{path}' not found. Ensure 'data.yaml' is in the working directory with correct train, val, and names fields.")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None, limit=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
        if limit:
            self.images = self.images[:limit]  # Limit dataset size for fast-dev-run

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = Image.open(img_path).convert("RGB")
        img = img.resize((640, 640))
        w, h = img.size

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    x_min = (x - bw / 2) * w
                    y_min = (y - bh / 2) * h
                    x_max = (x + bw / 2) * w
                    y_max = (y + bh / 2) * h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls) + 1)  # Add 1 because COCO expects 1-based labels
                    # Compute area of the bounding box
                    area = (x_max - x_min) * (y_max - y_min)
                    areas.append(area)
                    iscrowd.append(0)  # Assume single objects for license plates

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),  # Use idx as unique identifier
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def simple_evaluate(model, data_loader, device):
    """Simple evaluation without COCO metrics to avoid the assertion error"""
    model.eval()
    total_detections = 0
    total_images = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            
            # Get predictions (model in eval mode returns predictions, not losses)
            predictions = model(images)
            
            # Count detections for simple metric
            for pred in predictions:
                # Count detections with confidence > 0.5
                high_conf_detections = (pred['scores'] > 0.5).sum().item()
                total_detections += high_conf_detections
            
            total_images += len(images)
    
    # Return average detections per image as a simple metric
    avg_detections = total_detections / total_images if total_images > 0 else 0
    model.train()
    return avg_detections

def safe_mlflow_operation(operation, *args, **kwargs):
    """
    Safely execute MLflow operations with error handling
    """
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        print(f"MLflow operation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN License Plate Detection")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run in fast dev mode")
    parser.add_argument("--simple-eval", action="store_true", help="Use simple evaluation instead of COCO")
    parser.add_argument("--local-only", action="store_true", help="Skip MLflow logging and save locally only")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    
    # MLflow setup with error handling
    mlflow_enabled = not args.local_only
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri("http://ec2-13-215-137-215.ap-southeast-1.compute.amazonaws.com:5000")
            mlflow.set_experiment("LicensePlate-Detection")
            print("MLflow tracking enabled")
        except Exception as e:
            print(f"Failed to connect to MLflow server: {e}")
            print("Continuing with local logging only...")
            mlflow_enabled = False

    run_name = f"fasterrcnn-lp-detector-{int(time.time())}"
    output_dir = os.path.abspath(f"runs/fasterrcnn/{run_name}")
    os.makedirs(output_dir, exist_ok=True)

    data_yaml = load_data_yaml("data.yaml")
    train_img_dir = data_yaml["train"]
    val_img_dir = data_yaml["val"]
    train_lbl_dir = os.path.join(os.path.dirname(train_img_dir), "labels")
    val_lbl_dir = os.path.join(os.path.dirname(val_img_dir), "labels")
    class_names = data_yaml.get("names", [])
    num_classes = len(class_names) + 1  # Add 1 for background class in Faster R-CNN

    # Verify directories exist
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    # Limit dataset size for fast-dev-run
    dataset_limit = 10 if args.fast_dev_run else None
    dataset = YOLODataset(train_img_dir, train_lbl_dir, get_transform(), limit=dataset_limit)
    dataset_test = YOLODataset(val_img_dir, val_lbl_dir, get_transform(), limit=dataset_limit)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resnet18(weights="DEFAULT")
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                      aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    model.to(device)

    params = {
        "epochs": 10 if args.fast_dev_run else args.epochs,  # Fixed: was 50 for fast-dev-run
        "lr": 0.0025,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "batch_size": args.batch_size
    }

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=params["lr"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = torch.amp.GradScaler('cuda')

    # Start MLflow run with error handling
    mlflow_run = None
    if mlflow_enabled:
        try:
            mlflow_run = mlflow.start_run(run_name=run_name)
            safe_mlflow_operation(mlflow.log_params, params)
            safe_mlflow_operation(mlflow.set_tags, {
                "project": "License Plate Detection",
                "framework": "Faster R-CNN",
                "phase": "training",
                "num_classes": num_classes,
                "class_names": str(class_names)
            })
        except Exception as e:
            print(f"Failed to start MLflow run: {e}")
            mlflow_enabled = False

    print(f"Starting training for {params['epochs']} epochs...")
    
    try:
        for epoch in range(params["epochs"]):
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"\nEpoch {epoch + 1}/{params['epochs']}")
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=scaler)
            lr_scheduler.step()
            
            # Use simple evaluation or COCO evaluation based on flag
            if args.simple_eval or args.fast_dev_run:
                avg_detections = simple_evaluate(model, data_loader_test, device)
                if mlflow_enabled:
                    safe_mlflow_operation(mlflow.log_metric, "avg_detections_per_image", avg_detections, step=epoch)
                print(f"Epoch {epoch + 1}: Average detections per image = {avg_detections:.2f}")
            else:
                try:
                    coco_eval = evaluate(model, data_loader_test, device=device)
                    if coco_eval and coco_eval.coco_eval['bbox']:
                        stats = coco_eval.coco_eval['bbox'].stats
                        metrics = {
                            "AP": stats[0],
                            "AP50": stats[1],
                            "AP75": stats[2],
                            "APl": stats[5]
                        }
                        if mlflow_enabled:
                            safe_mlflow_operation(mlflow.log_metrics, metrics, step=epoch)
                        print(f"Epoch {epoch + 1}: AP={stats[0]:.3f}, AP50={stats[1]:.3f}")
                except Exception as e:
                    print(f"COCO evaluation failed: {e}")
                    print("Falling back to simple evaluation...")
                    avg_detections = simple_evaluate(model, data_loader_test, device)
                    if mlflow_enabled:
                        safe_mlflow_operation(mlflow.log_metric, "avg_detections_per_image", avg_detections, step=epoch)
                    print(f"Epoch {epoch + 1}: Average detections per image = {avg_detections:.2f}")

            if mlflow_enabled:
                log_system_metrics()

        # Save model locally
        model_path = os.path.join(output_dir, "fasterrcnn_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved locally to: {model_path}")

        # Try to upload to MLflow with retry mechanism
        if mlflow_enabled:
            upload_success = log_artifact_with_retry(model_path, artifact_path="models")
            if not upload_success:
                print(f"Model upload failed, but training completed successfully.")
                print(f"Model is available locally at: {model_path}")
        else:
            print("MLflow disabled - model saved locally only")
            
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        # End MLflow run safely
        if mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"Failed to end MLflow run: {e}")

    print("Training completed successfully!")

if __name__ == "__main__":
    main()