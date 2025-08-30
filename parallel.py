import torch
from models.create_fasterrcnn_model import return_fasterrcnn_resnet50_fpn_v2, return_fasterrcnn_resnet18_fpn, return_fasterrcnn_resnet34_fpn, return_fasterrcnn_resnet101_fpn
import time
from torchvision.transforms import functional as F
import torchvision.ops as ops
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
from utils.annotations import inference_annotations

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_thresh=0.001, method='linear'):
    """
    boxes: (N, 4) tensor [x1, y1, x2, y2]
    scores: (N,) tensor
    iou_threshold: IoU threshold for linear Soft-NMS
    sigma: variance for gaussian Soft-NMS
    score_thresh: drop boxes below this after decay
    method: 'linear' or 'gaussian'
    """
    if boxes.numel() == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    
    boxes = boxes.clone()
    scores = scores.clone()
    N = boxes.size(0)
    indices = torch.arange(N)
    keep_boxes = []
    keep_scores = []
    keep_indices = []

    for i in range(N):
        if scores.numel() == 0:
            break
            
        # Get max score box
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx].clone()
        max_score = scores[max_idx].clone()
        max_original_idx = indices[max_idx].clone()
        
        keep_boxes.append(max_box)
        keep_scores.append(max_score)
        keep_indices.append(max_original_idx)
        
        # Remove selected box from list
        boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]], dim=0)
        scores = torch.cat([scores[:max_idx], scores[max_idx+1:]], dim=0)
        indices = torch.cat([indices[:max_idx], indices[max_idx+1:]], dim=0)

        if boxes.numel() == 0:
            break

        # Compute IoU with the selected box
        x1 = torch.max(max_box[0], boxes[:, 0])
        y1 = torch.max(max_box[1], boxes[:, 1])
        x2 = torch.min(max_box[2], boxes[:, 2])
        y2 = torch.min(max_box[3], boxes[:, 3])
        
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area1 + area2 - inter)

        # Decay scores instead of discarding
        if method == 'linear':
            decay = torch.ones_like(iou)
            decay[iou > iou_threshold] -= iou[iou > iou_threshold]
            scores = scores * decay
        elif method == 'gaussian':
            scores = scores * torch.exp(- (iou * iou) / sigma)

        # Filter low scores
        keep_idx = scores > score_thresh
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        indices = indices[keep_idx]

    if len(keep_boxes) > 0:
        final_boxes = torch.stack(keep_boxes)
        final_scores = torch.stack(keep_scores)
        final_indices = torch.stack(keep_indices)
    else:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_indices = torch.empty((0,), dtype=torch.long)
    
    return final_boxes, final_scores, final_indices

# Custom function to extract RPN proposals
def extract_rpn_proposals(model, images, rpn_score_thresh=0.05, rpn_nms_thresh=0.7):
    """
    Extract RPN proposals from a model without running ROI heads
    
    Args:
        model: Faster R-CNN model
        images: List of image tensors
        rpn_score_thresh: RPN score threshold
        rpn_nms_thresh: RPN NMS threshold
        rpn_pre_nms_top_n: Number of proposals before NMS
        rpn_post_nms_top_n: Number of proposals after NMS
    
    Returns:
        proposals: List of proposal boxes for each image
        features: Backbone features for ROI pooling
    """
    model.eval()
    
    # Temporarily store original RPN parameters
    original_score_thresh = model.rpn.score_thresh
    original_nms_thresh = model.rpn.nms_thresh
    
    # Set custom RPN parameters
    model.rpn.score_thresh = rpn_score_thresh
    model.rpn.nms_thresh = rpn_nms_thresh
    
    with torch.no_grad():
        # Extract features from backbone
        features = model.backbone(images.tensors)
        
        # Run RPN to get proposals
        proposals, _ = model.rpn(images, features)
    
    # Restore original parameters
    model.rpn.score_thresh = original_score_thresh
    model.rpn.nms_thresh = original_nms_thresh
    
    return proposals, features

def run_roi_classification(model, images, proposals, features):
    """
    Run ROI heads on merged proposals using one model's classifier
    
    Args:
        model: Faster R-CNN model (we'll use its ROI heads)
        images: ImageList object
        proposals: List of proposal boxes
        features: Backbone features
    
    Returns:
        detections: Final detection results
    """
    model.eval()
    
    with torch.no_grad():
        # Run ROI heads on merged proposals
        detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes)
        detections = model.transform.postprocess(detections, images.image_sizes, original_size)
    
    return detections

def merge_proposals(proposals_list):
    """
    Merge proposals from multiple models
    
    Args:
        proposals_list: List of proposal lists from different models
    
    Returns:
        merged_proposals: List of merged proposals for each image
    """
    num_images = len(proposals_list[0])
    merged_proposals = []
    
    for img_idx in range(num_images):
        # Collect all proposals for this image
        all_proposals = []
        for model_proposals in proposals_list:
            all_proposals.append(model_proposals[img_idx])
        
        # Concatenate proposals from all models
        if len(all_proposals) > 0:
            merged_boxes = torch.cat(all_proposals, dim=0)
            merged_proposals.append(merged_boxes)
        else:
            # No proposals for this image
            merged_proposals.append(torch.empty((0, 4)))
    
    return merged_proposals

# Load models
model1 = return_fasterrcnn_resnet18_fpn(num_classes=6)
model1.load_state_dict(torch.load('outputs/resnet18.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model1.eval()

model2 = return_fasterrcnn_resnet34_fpn(num_classes=6)
model2.load_state_dict(torch.load('outputs/resnet34.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model2.eval()

model3 = return_fasterrcnn_resnet50_fpn_v2(num_classes=6)
model3.load_state_dict(torch.load('outputs/resnet50v2.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model3.eval()

model4 = return_fasterrcnn_resnet101_fpn(num_classes=6)
model4.load_state_dict(torch.load('outputs/resnet101.pth', map_location=torch.device('cpu'), weights_only=False)['model_state_dict'])
model4.eval()

models = [model1, model2, model3, model4]

CLASS_NAMES = [
    '__background__',
    'text_overlap',
    'component_occlusion',
    'missing_image',
    'null_text',
    'element_overflow'
]

DETECTION_THRESHOLD = 0.7
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

# Image processing
image_folder = "./images"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
os.makedirs("Results", exist_ok=True)
start_time = time.time()

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb)
    
    # Prepare ImageList (required by Faster R-CNN)
    from torchvision.models.detection.image_list import ImageList
    original_size = [(image_tensor.shape[1], image_tensor.shape[2])]  # (H, W)
    images, _ = model4.transform([image_tensor], None)  # use primary model's transform
    
    # Step 1: Extract RPN proposals from all models in parallel
    def extract_proposals_parallel(model):
        return extract_rpn_proposals(
            model, images, 
            rpn_score_thresh=0.05,
            rpn_nms_thresh=0.7,
        )
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_proposals_parallel, model) for model in models]
        results = [f.result() for f in futures]
    
    # Extract proposals and features
    proposals_list = [result[0] for result in results]
    features_list = [result[1] for result in results]
    
    # Step 2: Merge proposals from all models
    merged_proposals = merge_proposals(proposals_list)
    
    # Step 3: Use one model's backbone features and ROI heads for classification
    # We'll use the ResNet101 model (model4) as the primary classifier since it's the most powerful
    primary_model = model4
    primary_features = features_list[3]  # Features from ResNet101
    
    # Run classification on merged proposals
    final_detections = run_roi_classification(
        primary_model, images, merged_proposals, primary_features
    )
    
    # Step 4: Apply final NMS and filtering
    if len(final_detections) > 0 and len(final_detections[0]['boxes']) > 0:
        boxes = final_detections[0]['boxes']
        scores = final_detections[0]['scores']
        labels = final_detections[0]['labels']
        
        # Apply final NMS
        if len(boxes) > 0:
            final_boxes, final_scores, keep_indices = soft_nms(boxes, scores, iou_threshold=0.5)
            final_labels = labels[keep_indices]
        else:
            final_boxes = torch.empty((0, 4))
            final_scores = torch.empty((0,))
            final_labels = torch.empty((0,), dtype=torch.long)
    else:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)
    
    # Prepare outputs for annotation
    outputs_dict = [{
        'boxes': final_boxes,
        'scores': final_scores,
        'labels': final_labels
    }]
    
    # Step 5: Annotate and save
    result_img = inference_annotations(
        outputs_dict, DETECTION_THRESHOLD, CLASS_NAMES, COLORS, image_cv.copy()
    )
    
    # Print detection details
    num_detections = len(final_boxes)
    print(f"\n{image_name}: Found {num_detections} detections above threshold {DETECTION_THRESHOLD}")
    
    if num_detections > 0:
        # Filter detections above threshold for display
        above_threshold = final_scores >= DETECTION_THRESHOLD
        display_boxes = final_boxes[above_threshold]
        display_scores = final_scores[above_threshold]
        display_labels = final_labels[above_threshold]
        
        for i in range(len(display_boxes)):
            box = display_boxes[i]
            score = display_scores[i].item()
            label_idx = display_labels[i].item()
            class_name = CLASS_NAMES[label_idx]
            print(f"  Detection {i+1}: {class_name} (confidence: {score:.3f}) "
                  f"[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    save_path = os.path.join("Results", f"rpn_ensemble_{image_name}")
    cv2.imwrite(save_path, result_img)
    
    print(f"Saved annotated image to {save_path}\n")

end_time = time.time()
print(f"RPN-level ensemble inference took {end_time - start_time:.2f} seconds")