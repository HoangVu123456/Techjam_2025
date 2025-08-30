import torch
from models.create_fasterrcnn_model import return_fasterrcnn_resnet50_fpn_v2, return_fasterrcnn_resnet18_fpn, return_fasterrcnn_resnet34_fpn, return_fasterrcnn_resnet101_fpn
import time
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import torchvision.ops as ops
import os

# 1. Load models
model1 = return_fasterrcnn_resnet18_fpn(num_classes=6)
model1.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet18.pth', weights_only=False)['model_state_dict'])
model1.eval()

model2 = return_fasterrcnn_resnet34_fpn(num_classes=6)
model2.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet34.pth', weights_only=False)['model_state_dict'])
model2.eval()

models = [model1, model2]

# 2. Loop through all images in a folder
image_folder = r"F:\fasterrcnn_resnet50_fpn_v2_new_dataset\inputs"  # <-- change this to your folder
os.makedirs("Results", exist_ok=True)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
start_time = time.time()

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)

    # 3. Run inference (concurrent/sequential)
    outputs = []
    for model in models:
        with torch.no_grad():
            output = model([image_tensor])[0]
            outputs.append(output)

    # 4. Ensemble predictions (NMS example)
    all_boxes = torch.cat([o['boxes'] for o in outputs])
    all_scores = torch.cat([o['scores'] for o in outputs])
    all_labels = torch.cat([o['labels'] for o in outputs])

    keep = ops.nms(all_boxes, all_scores, iou_threshold=0.5)
    ensemble_boxes = all_boxes[keep]
    ensemble_scores = all_scores[keep]
    ensemble_labels = all_labels[keep]

    print(f"{image_name}: Ensembled predictions:", ensemble_boxes, ensemble_scores, ensemble_labels)

    # 5. Save result image
    image_draw = image.convert("RGB")
    draw = ImageDraw.Draw(image_draw)
    for box, score, label in zip(ensemble_boxes, ensemble_scores, ensemble_labels):
        box = box.tolist()
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{int(label.item())}:{score.item():.2f}", fill="yellow")
    save_path = os.path.join("Results", f"ensemble_{image_name}")
    image_draw.save(save_path)

end_time = time.time()
print(f"Inference took {end_time - start_time:.2f} seconds")