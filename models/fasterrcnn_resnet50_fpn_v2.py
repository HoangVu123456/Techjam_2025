import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    """
    Create a Faster R-CNN model from scratch (no COCO pretrained weights).
    Args:
        num_classes (int): Number of classes including background.
    """
    # Load Faster R-CNN with randomly initialized weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)

    # Replace the head for the number of classes you need
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    # Example: 81 classes including background
    model = create_model(num_classes=6)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params:,} total parameters.")
    print(f"{total_trainable_params:,} training parameters.")
