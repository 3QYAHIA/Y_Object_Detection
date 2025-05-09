#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import json
import time
from PIL import Image
from torchvision import transforms

# Import project modules
from models.detector import get_faster_rcnn_model
from utils.visualization import plot_image_with_boxes
from data.coco_dataset import CocoDataset

def load_category_names(ann_file):
    """
    Load category names from COCO annotation file
    
    Args:
        ann_file: Path to annotation file
        
    Returns:
        category_names: Dictionary mapping category ID to name
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    categories = {}
    for cat in data['categories']:
        categories[cat['id']] = cat['name']
    
    return categories

def load_image(image_path):
    """
    Load and preprocess image for model input
    
    Args:
        image_path: Path to image file
        
    Returns:
        image: Processed image tensor
        original_image: Original image for visualization
    """
    # Read image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(original_image)
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(pil_image)
    
    return image, original_image

def run_inference(model, image, device, threshold=0.5):
    """
    Run inference on a single image
    
    Args:
        model: Detection model
        image: Input image tensor
        device: Device to run inference on
        threshold: Confidence threshold for detections
        
    Returns:
        filtered_boxes: Filtered bounding boxes
        filtered_labels: Filtered class labels
        filtered_scores: Filtered confidence scores
    """
    # Move image to device
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model([image])[0]
    
    # Get boxes, labels and scores
    boxes = prediction['boxes'].cpu()
    labels = prediction['labels'].cpu()
    scores = prediction['scores'].cpu()
    
    # Filter by confidence
    mask = scores >= threshold
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]
    
    return filtered_boxes, filtered_labels, filtered_scores

def main(args):
    # Set up output directory
    output_dir = os.path.join("outputs", args.backbone, "detections")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class names
    ann_file = args.ann_file
    if not os.path.exists(ann_file):
        # Try to find annotation file in default location
        data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Prefer tiny_subset
        tiny_subset_path = os.path.join(data_root, "coco", "tiny_subset", "annotations", "instances_val2017.json")
        subset_path = os.path.join(data_root, "coco", "subset", "annotations", "instances_train2017.json")
        default_path = os.path.join(data_root, "coco", "annotations", "instances_val2017.json")
        
        if os.path.exists(tiny_subset_path) and args.tiny:
            ann_file = tiny_subset_path
            print("Using tiny subset annotations (5 classes)")
        elif os.path.exists(subset_path) and args.subset:
            ann_file = subset_path
            print("Using subset annotations (10 classes)")
        elif os.path.exists(default_path):
            ann_file = default_path
            print("Using full COCO annotations")
        else:
            raise FileNotFoundError(f"No annotation file found at {args.ann_file} or in default locations")
    
    # Load category names
    print(f"Loading category names from {ann_file}")
    category_names = load_category_names(ann_file)
    num_classes = len(category_names) + 1  # +1 for background
    
    # Create class name list
    class_names = []
    for i in range(1, num_classes):  # Start from 1 to skip background
        class_names.append("unknown")
    
    for cat_id, cat_name in category_names.items():
        if cat_id < len(class_names) + 1:  # +1 because class_names starts from 1
            class_names[cat_id - 1] = cat_name
    
    print(f"Loaded {len(class_names)} class names: {', '.join(class_names)}")
    
    # Create model
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=False,
        trainable_backbone_layers=0
    )
    
    # Load model weights
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find checkpoint in default location
        default_path = os.path.join("outputs", args.backbone, "best_model.pth")
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or {default_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Move model to device
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process image or directory
    if os.path.isdir(args.input):
        # Process all images in directory
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        # Process single image
        image_files = [args.input]
    
    print(f"Processing {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        
        # Load image
        image, original_image = load_image(image_file)
        
        # Run inference
        start_time = time.time()
        boxes, labels, scores = run_inference(model, image, device, args.threshold)
        inference_time = time.time() - start_time
        
        print(f"  Detected {len(boxes)} objects in {inference_time:.3f} seconds")
        
        # Print detected objects
        for j in range(len(boxes)):
            label_idx = labels[j].item()
            class_name = class_names[label_idx - 1] if label_idx > 0 and label_idx <= len(class_names) else "unknown"
            score = scores[j].item()
            print(f"  - {class_name}: {score:.3f}")
        
        # Create visualization
        fig = plot_image_with_boxes(
            img=original_image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_names=class_names
        )
        
        # Save visualization
        output_name = os.path.basename(image_file)
        output_path = os.path.join(output_dir, f"detection_{output_name}")
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved detection to {output_path}")
    
    print("Detection complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection on images")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                       help="Path to model checkpoint")
    
    # Input parameters
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--ann-file", type=str, default="annotations.json",
                       help="Path to annotation file with category names")
    parser.add_argument("--subset", action="store_true",
                       help="Use subset of COCO dataset (10 classes)")
    parser.add_argument("--tiny", action="store_true", default=True,
                       help="Use tiny subset of COCO dataset (5 classes, <300MB)")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 