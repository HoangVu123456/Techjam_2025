import torchvision
from torchvision.models import ResNet34_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def create_model(num_classes, pretrained=True, coco_model=False):
    # Create a ResNet-34 FPN backbone
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    backbone = resnet_fpn_backbone('resnet34', weights=weights)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

if __name__ == '__main__':
    model = create_model(num_classes=81, pretrained=True)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")