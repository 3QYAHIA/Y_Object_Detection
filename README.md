# Object Detection with CNN Architectures

This project implements and compares different CNN backbones for object detection using PyTorch and Faster R-CNN.

## Features
- Implements object detection using COCO dataset
- Compares two CNN backbones (ResNet-50 and MobileNetV2) on performance and speed
- Evaluates models using mAP, precision, recall and F1-score
- Visualizes detection results with bounding boxes

## Setup and Installation

```bash
# Clone repository
git clone https://github.com/your-username/object-detection.git
cd object-detection

# Install dependencies
pip install -r requirements.txt

# Download and prepare COCO dataset
python data/download_coco.py

# Train models
python train.py --backbone resnet50
python train.py --backbone mobilenet_v2

# Evaluate models
python evaluate.py --backbone resnet50
python evaluate.py --backbone mobilenet_v2

# Run inference
python detect.py --backbone resnet50 --image path/to/image.jpg
```

## Project Structure
- `data/`: Dataset handling and processing
- `models/`: Model architectures 
- `utils/`: Utility functions
- `evaluation/`: Evaluation metrics and visualization
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `detect.py`: Inference script
- `compare_backbones.py`: Backbone comparison script

## Collaboration

This project is open for collaboration. Please follow these guidelines:

1. Create a feature branch for your changes: `git checkout -b feature/your-feature-name`
2. Make your changes and test them
3. Submit a pull request with a clear description of your changes

## Future Improvements

- Add more backbone options (MobileNetV3, EfficientNet)
- Implement YOLOv5 and SSD models for comparison
- Improve training speed with mixed precision
- Add support for custom datasets

## License
MIT 