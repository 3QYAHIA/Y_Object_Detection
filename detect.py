#!/usr/bin/env python3
# Set matplotlib backend to non-interactive to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import torch
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
from torchvision import transforms
import glob

# Import project modules
from models.detector import get_faster_rcnn_model
from utils.visualization import draw_boxes_on_image

def load_model(backbone, checkpoint_path, device):
    """
    Load model from checkpoint
    
    Args:
        backbone: Backbone name (resnet50 or mobilenet_v2)
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded model
    """
    # Create model - COCO has 91 classes (including background)
    model = get_faster_rcnn_model(
        num_classes=91,
        backbone=backbone,
        pretrained=False,
        trainable_backbone_layers=0
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find checkpoint in default location
        default_path = os.path.join("outputs", backbone, "best_model.pth")
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
    
    return model

def process_image(image_path, transform):
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image
        transform: Transform to apply
        
    Returns:
        img: Preprocessed image
        orig_img: Original image
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Save original image for visualization
    orig_img = np.array(img)
    
    # Apply transformations
    img = transform(img)
    
    return img, orig_img

def run_inference(model, image, device):
    """
    Run inference on a single image
    
    Args:
        model: Model to use for inference
        image: Image to run inference on
        device: Device to run on
        
    Returns:
        output: Dictionary with boxes, labels, scores
    """
    # Move image to device
    image = image.to(device)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        output = model(image)[0]
        inference_time = time.time() - start_time
    
    return output, inference_time

def get_class_names():
    """
    Get COCO class names
    
    Returns:
        class_names: Dictionary mapping class indices to names
    """
    class_names = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    return class_names

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.backbone, args.checkpoint, device)
    
    # Create output directory
    output_dir = args.output
    if not output_dir:
        output_dir = os.path.join("outputs", args.backbone, "detections")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get class names
    class_names = get_class_names()
    
    # Process input
    image_paths = []
    if os.path.isdir(args.input):
        # Process all images in directory
        extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        # Process single image
        image_paths = [args.input]
    
    # Check if any images were found
    if not image_paths:
        print(f"No images found at {args.input}")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Run inference on each image
    total_inference_time = 0
    detection_count = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Load and preprocess image
        img, orig_img = process_image(image_path, transform)
        
        # Run inference
        output, inference_time = run_inference(model, img, device)
        total_inference_time += inference_time
        
        # Filter detections by confidence
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= args.conf_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        detection_count += len(boxes)
        
        # Convert labels to class names
        class_labels = [class_names.get(label, f"Unknown ({label})") for label in labels]
        
        # Draw boxes on image
        result_img = draw_boxes_on_image(orig_img, boxes, class_labels, scores)
        
        # Save result
        output_path = os.path.join(output_dir, f"detection_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        print(f"  Found {len(boxes)} objects in {inference_time:.3f}s")
        print(f"  Saved result to {output_path}")
    
    # Print summary
    avg_time = total_inference_time / len(image_paths)
    avg_fps = 1 / avg_time if avg_time > 0 else 0
    
    print("\nDetection Summary:")
    print(f"  Processed {len(image_paths)} images")
    print(f"  Found {detection_count} objects")
    print(f"  Average inference time: {avg_time:.3f}s per image")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection on images")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                       help="Path to model checkpoint")
    
    # Input/output parameters
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output directory")
    
    # Detection parameters
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 