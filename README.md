# Bug404: AI-Powered UI Bug Detection

Bug404 is an AI model that automatically detects **unexpected UI anomalies** from screenshots. Unlike traditional UI testing pipelines, which rely on scripted test cases, Bug404 can find defects that testers might not anticipate — for example, a missing image elsewhere on the page, even if the main functionality works. By using object detection, Bug404 adds an extra layer of confidence to UI automation.

---

## Features
- Detects **UI bugs beyond scripted tests**
- Multi-scale feature extraction for better coverage
- Parallel inference for fast runtime
- Soft Non-Maximum Suppression (NMS) to keep the most confident detections

---

## How It Works
Bug404 is built on **Faster R-CNN** with multiple ResNet backbones: ResNet-18, ResNet-34, and ResNet-50 for feature extraction, and ResNet-101 for classification. Key steps:

1. **Data Preparation:** Raw screenshots are split, annotated, and transformed for training.
2. **Model Creation:** Faster R-CNN is instantiated with multiple backbone options.
3. **Region Proposals:** A Region Proposal Network (RPN) generates candidate bounding boxes in parallel.
4. **Prediction:** Proposals are merged and filtered using Soft NMS to retain the most confident detections.
5. **Evaluation:** COCO metrics and PyTorch utilities are used for training and testing performance.

---

## Get started

### Installation

```bash
# Clone the repo
git clone https://github.com/hoangvu123456/techjam_2025.git
cd techjam_2025

# Create and activate virtual environment
conda create -n bug404 python=3.10
conda activate bug404

# Install dependencies
pip install -r requirements.txt

```
### Download pretrained model

```bash
# Download trained model from this link
# https://drive.google.com/drive/folders/1tB5cQR1bWVDG0dEhHR1PH2EnVlhznmvx
# Put it inside the outputs/ folder
```
---

### Usage

To try model on your own images:

```bash
# Bug404 expect input inside the images/ folder.
# Run parallel.py
python parallel.py
```

## Train your own model

```bash
# Training with Faster RCNN ResNet50 FPN v2 model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn_v2 --epochs 2 --config data_configs/voc.yaml --no-mosaic --batch-size 4

# Training on ResNet50 FPN v2 with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn_v2 --epochs 2 --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4

# Training on ResNet50 FPN v2 with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn_v2 --epochs 2 --use-train-aug --config data_configs/voc.yaml --project-name resnet50fpn_voc --batch-size 4
```
