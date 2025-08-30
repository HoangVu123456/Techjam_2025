import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt
from torchvision.ops import nms

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images    

def apply_post_nms(boxes, scores, labels, iou_thresh, class_agnostic=False):
    """
    Apply post-processing NMS to detection outputs.
    
    Args:
        boxes: Tensor [N, 4] - bounding boxes
        scores: Tensor [N] - confidence scores  
        labels: Tensor [N] - class labels
        iou_thresh: float - IoU threshold for NMS
        class_agnostic: bool - if True, apply NMS across all classes
        
    Returns:
        Filtered boxes, scores, labels tensors
    """
    if boxes.numel() == 0:
        return boxes, scores, labels
        
    if class_agnostic:
        # Apply NMS across all classes
        keep = nms(boxes, scores, iou_thresh)
    else:
        # Apply NMS per class separately
        keep_idxs = []
        for c in labels.unique():
            cls_mask = labels == c
            if cls_mask.sum() == 0:
                continue
            cls_keep = nms(boxes[cls_mask], scores[cls_mask], iou_thresh)
            # Map back to original indices
            original_indices = cls_mask.nonzero(as_tuple=True)[0]
            keep_idxs.append(original_indices[cls_keep])
        
        if len(keep_idxs) > 0:
            keep = torch.cat(keep_idxs)
        else:
            keep = torch.tensor([], dtype=torch.long)
    
    # Sort by score descending for consistent ordering
    if len(keep) > 0:
        keep = keep[scores[keep].argsort(descending=True)]
    
    return boxes[keep], scores[keep], labels[keep]

def inference_annotations_filtered(boxes, scores, labels, classes, colors, orig_image):
    """
    Draw annotations using pre-filtered detection results.
    
    Args:
        boxes: Tensor [N, 4] - filtered bounding boxes
        scores: Tensor [N] - filtered confidence scores
        labels: Tensor [N] - filtered class labels
        classes: list - class names
        colors: array - colors for each class
        orig_image: ndarray - original image to annotate
        
    Returns:
        Annotated image
    """
    if boxes.numel() == 0:
        return orig_image
        
    boxes_np = boxes.to(torch.int32).numpy()
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Get predicted class names
    pred_classes = [classes[i] for i in labels_np]
    
    lw = max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width
    tf = max(lw - 1, 1)  # Font thickness
    
    # Draw bounding boxes and class names
    for j, box in enumerate(boxes_np):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        class_name = pred_classes[j]
        score = scores_np[j]
        color = colors[classes.index(class_name)]
        
        # Draw rectangle
        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        
        # Prepare label with confidence score
        label = f"{class_name}: {score:.2f}"
        
        # Calculate text size
        w, h = cv2.getTextSize(
            label, 
            0, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]
        
        outside = p1[1] - h >= 3
        p2_text = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        
        # Draw text background
        cv2.rectangle(
            orig_image, 
            p1, 
            p2_text, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )
        
        # Draw text
        cv2.putText(
            orig_image, 
            label, 
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3.8, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    
    return orig_image

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '-c', '--config', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--post-nms-iou', type=float, default=None,
        help='optional extra NMS IoU threshold after model outputs (e.g., 0.5)'
    )
    parser.add_argument(
        '--class-agnostic-nms', action='store_true',
        help='apply post NMS across all classes jointly instead of per-class'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    DEVICE = args['device']
    OUT_DIR = set_infer_dir()

    # Load the pretrained model
    if args['weights'] is None:
        # If the config file is still None, 
        # then load the default one for COCO.
        if data_configs is None:
            with open(os.path.join('data_configs', 'test_image_config.yaml')) as file:
                data_configs = yaml.safe_load(file)
            NUM_CLASSES = data_configs['NC']
            CLASSES = data_configs['CLASSES']
        try:
            build_model = create_model[args['model']]
        except:
            build_model = create_model['fasterrcnn_resnet50_fpn']
        model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['config']['NC']
            CLASSES = checkpoint['config']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[str(args['model'])]
        except:
            build_model = create_model[checkpoint['model_name']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    if args['input'] == None:
        DIR_TEST = data_configs['image_path']
        test_images = collect_all_images(DIR_TEST)
    else:
        DIR_TEST = args['input']
        test_images = collect_all_images(DIR_TEST)
    print(f"Test instances: {len(test_images)}")

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = args['threshold']

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    for i in range(len(test_images)):
        # Get the image file name for saving output later on.
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        # Extract detection components
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores'] 
        labels = outputs[0]['labels']
        
        # Apply score filtering
        score_mask = scores >= detection_threshold
        boxes = boxes[score_mask]
        scores = scores[score_mask] 
        labels = labels[score_mask]
        
        # Apply optional post-NMS if requested
        if args['post_nms_iou'] is not None and boxes.numel() > 0:
            boxes, scores, labels = apply_post_nms(
                boxes, scores, labels,
                iou_thresh=args['post_nms_iou'],
                class_agnostic=args['class_agnostic_nms']
            )
        
        # Carry further only if there are detected boxes after all filtering
        if boxes.numel() > 0:
            orig_image = inference_annotations_filtered(
                boxes, scores, labels, CLASSES, COLORS, orig_image
            )
            if args['show_image']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {i+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)