# Object Detection with CNN Backbones

A simple implementation of object detection using Faster R-CNN with different CNN backbones (ResNet-50 and MobileNetV2) on the Pascal VOC dataset.

## Features

- Implementation of Faster R-CNN framework with PyTorch
- Support for multiple CNN backbones:
  - ResNet-50: Higher accuracy, more parameters
  - MobileNetV2: Faster inference, fewer parameters
- Support for Pascal VOC dataset: 20 classes, standard benchmark
- Automatic dataset download and preprocessing
- Comprehensive evaluation metrics:
  - mAP at IoU 0.5 and 0.75
  - Precision, Recall, F1-score per class
  - Confusion matrix
  - Detection visualizations

## Setup and Installation

```bash
# Clone repository
git clone https://github.com/your-username/object-detection.git
cd object-detection

# Install dependencies
pip install -r requirements.txt
```

## Training

### Pascal VOC Dataset

```bash
# Train with Pascal VOC 2012 dataset
python train.py --voc-year 2012 --backbone resnet50 --download

# Train with Pascal VOC 2007 dataset
python train.py --voc-year 2007 --backbone resnet50 --download

# Use trainval split and more epochs for better accuracy
python train.py --voc-year 2012 --voc-train-set trainval --epochs 20 --backbone resnet50 --download
```

Key parameters:
- `--backbone`: Model backbone (resnet50, mobilenet_v2)
- `--voc-year`: For Pascal VOC, choose dataset year (2007-2012)
- `--voc-train-set`: For Pascal VOC, choose training set (train or trainval)
- `--voc-val-set`: For Pascal VOC, choose validation set (val or test, test only for 2007)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--download`: Download dataset if not found

## Evaluation

The evaluation script calculates all required metrics:

```bash
# Evaluate ResNet-50 model on Pascal VOC
python evaluate.py --backbone resnet50

# Compare backbones on specific metrics
python compare_backbones.py
```

## Inference

```bash
# Run inference on images with Pascal VOC model
python detect.py --backbone resnet50 --input path/to/image.jpg
```

## All-in-One Script

For convenience, you can use the `main.py` script to run all steps:

```bash
# Pascal VOC Dataset
python main.py --voc-year 2012 --download --train --evaluate --compare
```

## Project Structure

- `train.py`: Training script with data loading and model training
- `evaluate.py`: Evaluation script for calculating metrics (mAP, precision/recall, etc.)
- `detect.py`: Inference script for running on new images
- `compare_backbones.py`: Script to compare ResNet-50 vs MobileNetV2 on speed and accuracy
- `main.py`: All-in-one script for the complete workflow
- `models/detector.py`: Faster R-CNN model with different CNN backbones
- `data/voc_dataset.py`: Pascal VOC dataset handling
- `utils/visualization.py`: Functions to visualize detection results

## Dataset Information

### Pascal VOC Dataset
The Pascal VOC (Visual Object Classes) dataset contains 20 object categories and is a standard benchmark for object detection. It's available in several versions (2007-2012).

The VOC categories are:
- aeroplane, bicycle, bird, boat, bottle
- bus, car, cat, chair, cow
- diningtable, dog, horse, motorbike, person
- pottedplant, sheep, sofa, train, tvmonitor

## How It Works

This project implements Faster R-CNN, a two-stage object detection model:
1. **Backbone**: Extracts features from images (ResNet-50 or MobileNetV2)
2. **Region Proposal Network (RPN)**: Generates region proposals
3. **RoI Pooling**: Extracts features for each proposal
4. **Classifier**: Classifies each region and refines bounding boxes

The training process:
1. Load dataset with automatic download if needed
2. Initialize model with selected backbone
3. Train for specified number of epochs
4. Save best model based on validation accuracy
5. Create visualizations to verify model performance

## License
MIT 
