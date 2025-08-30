from models import *
from models import fasterrcnn_resnet101_fpn
from models import fasterrcnn_resnet18_fpn
from models import fasterrcnn_resnet34_fpn

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet50_fpn_v2(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn_v2.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet101_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet101_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet18_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet18_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet34_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet34_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': return_fasterrcnn_resnet50_fpn,
    'fasterrcnn_resnet50_fpn_v2': return_fasterrcnn_resnet50_fpn_v2,
    'fasterrcnn_resnet101_fpn': return_fasterrcnn_resnet101_fpn,
    'fasterrcnn_resnet34_fpn': return_fasterrcnn_resnet34_fpn,
    'fasterrcnn_resnet18_fpn': return_fasterrcnn_resnet18_fpn
}