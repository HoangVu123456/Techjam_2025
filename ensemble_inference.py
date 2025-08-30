import torch
from models.create_fasterrcnn_model import return_fasterrcnn_resnet50_fpn_v2, return_fasterrcnn_resnet18_fpn, return_fasterrcnn_resnet34_fpn, return_fasterrcnn_resnet101_fpn

from torchvision.transforms import functional as F
from PIL import Image
import torchvision.ops as ops

# 1. Load models
model1 = return_fasterrcnn_resnet18_fpn(num_classes=6)
model1.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet18.pth', weights_only=False)['model_state_dict'])
model1.eval()

model2 = return_fasterrcnn_resnet34_fpn(num_classes=6)
model2.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet34.pth', weights_only=False)['model_state_dict'])
model2.eval()

model3 = return_fasterrcnn_resnet50_fpn_v2(num_classes=6)
model3.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet50.pth', weights_only=False)['model_state_dict'])
model3.eval()

model4 = return_fasterrcnn_resnet101_fpn(num_classes=6)
model4.load_state_dict(torch.load('outputs/training/fasterrcnn_resnet50_fpn_v2_ppe/model_resnet101.pth', weights_only=False)['model_state_dict'])
model4.eval()

# 2. Load and preprocess image
image = Image.open('bug.5087.jpg')
image_tensor = F.to_tensor(image)

# 3. Run inference
outputs = []
for model in [model1, model2, model3, model4]:
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

print("Ensembled predictions:", ensemble_boxes, ensemble_scores, ensemble_labels)