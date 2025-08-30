import torch
from models.create_fasterrcnn_model import return_fasterrcnn_resnet50_fpn_v2, return_fasterrcnn_resnet18_fpn, return_fasterrcnn_resnet34_fpn, return_fasterrcnn_resnet101_fpn
import time
from torchvision.transforms import functional as F
import torchvision.ops as ops
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
from utils.annotations import inference_annotations

# 1. Load models
model1 = return_fasterrcnn_resnet18_fpn(num_classes=6)
model1.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/resnet18.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model1.eval()

model2 = return_fasterrcnn_resnet34_fpn(num_classes=6)
model2.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/resnet34.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model2.eval()

model3 = return_fasterrcnn_resnet50_fpn_v2(num_classes=6)
model3.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/resnet50.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model3.eval()

models = [model1, model2, model3]

CLASS_NAMES = [
    '__background__',
    'text_overlap',
    'component_occlusion',
    'missing_image',
    'null_text',
    'element_overflow'
]

DETECTION_THRESHOLD = 0.8  # Set your threshold
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

# 2. Loop through all images in a folder
image_folder = r"F:\fasterrcnn_resnet50_fpn_v2_new_dataset\inputs\images"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
os.makedirs("Results", exist_ok=True)
start_time = time.time()

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    # Read with OpenCV (BGR)
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb)

    # 3. Run inference in parallel
    def run_model(model, image_tensor):
        with torch.no_grad():
            return model([image_tensor])[0]

    # Define different thresholds for each model
    threshold_18 = 0.9
    threshold_34 = 0.8
    threshold_50 = 0.7

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_model, model, image_tensor) for model in models]
        outputs = [f.result() for f in futures]

    # Filter outputs by model-specific thresholds
    filtered_outputs = []
    
    # Filter ResNet18 outputs
    if len(outputs[0]['scores']) > 0:
        mask_18 = outputs[0]['scores'] > threshold_18
        if mask_18.any():
            filtered_18 = {
                'boxes': outputs[0]['boxes'][mask_18],
                'scores': outputs[0]['scores'][mask_18],
                'labels': outputs[0]['labels'][mask_18]
            }
            filtered_outputs.append(filtered_18)
    
    # Filter ResNet34 outputs
    if len(outputs[1]['scores']) > 0:
        mask_34 = outputs[1]['scores'] > threshold_34
        if mask_34.any():
            filtered_34 = {
                'boxes': outputs[1]['boxes'][mask_34],
                'scores': outputs[1]['scores'][mask_34],
                'labels': outputs[1]['labels'][mask_34]
            }
            filtered_outputs.append(filtered_34)
    
    # Filter ResNet50 outputs
    if len(outputs[2]['scores']) > 0:
        mask_50 = outputs[2]['scores'] > threshold_50
        if mask_50.any():
            filtered_50 = {
                'boxes': outputs[2]['boxes'][mask_50],
                'scores': outputs[2]['scores'][mask_50],
                'labels': outputs[2]['labels'][mask_50]
            }
            filtered_outputs.append(filtered_50)

    # 4. Ensemble predictions (NMS example)
    if len(filtered_outputs) == 0:
        # No detections above threshold, create empty tensors
        ensemble_boxes = torch.empty((0, 4))
        ensemble_scores = torch.empty((0,))
        ensemble_labels = torch.empty((0,), dtype=torch.long)
    else:
        all_boxes = torch.cat([o['boxes'] for o in filtered_outputs])
        all_scores = torch.cat([o['scores'] for o in filtered_outputs])
        all_labels = torch.cat([o['labels'] for o in filtered_outputs])

        keep = ops.nms(all_boxes, all_scores, iou_threshold=0.5)
        ensemble_boxes = all_boxes[keep]
        ensemble_scores = all_scores[keep]
        ensemble_labels = all_labels[keep]

    # Prepare outputs dict as expected by inference_annotations
    outputs_dict = [{
        'boxes': ensemble_boxes,
        'scores': ensemble_scores,
        'labels': ensemble_labels
    }]

    # 5. Annotate using your utility (OpenCV)
    result_img = inference_annotations(
        outputs_dict, DETECTION_THRESHOLD, CLASS_NAMES, COLORS, image_cv.copy()
    )

    save_path = os.path.join("Results", f"ensemble_{image_name}")
    cv2.imwrite(save_path, result_img)

    print(f"{image_name}: Saved to {save_path}")

end_time = time.time()
print(f"Parallel inference took {end_time - start_time:.2f} seconds")