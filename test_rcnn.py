import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet18
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import torchvision

# CONFIG
IMAGE_PATH = "car3.jpg"  # เปลี่ยนเป็น path รูปของคุณ
MODEL_PATH = r"runs/fasterrcnn/fasterrcnn-lp-detector-1748531370/fasterrcnn_model.pt" #1748459263, 1748463367, 1748500118
CLASS_NAMES = ['license_plate']

# BUILD MODEL
def get_model(num_classes):
    backbone = resnet18(weights="DEFAULT")
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

# LOAD MODEL
def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# DRAW PREDICTIONS
def draw_boxes(image, boxes, scores, labels, threshold=0.5):
    fig, ax = plt.subplots(1)
    ax.imshow(image, aspect='equal')  # aspect='equal' รักษาสัดส่วนภาพ

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 10,
                f"{CLASS_NAMES[label-1]}: {score:.2f}",
                color='white', backgroundcolor='red', fontsize=10)
    plt.axis('off')
    plt.show()

# MAIN
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES) + 1).to(device)

    # โหลดรูปตามขนาดจริงเลย (ไม่ resize)
    image = Image.open(IMAGE_PATH).convert("RGB")

    # แปลงเป็น tensor แล้วส่งให้ model
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    # วาดผลลัพธ์บนรูปเดิม (original size)
    draw_boxes(image, prediction["boxes"].cpu(),
               prediction["scores"].cpu(),
               prediction["labels"].cpu(),
               threshold=0.5)
