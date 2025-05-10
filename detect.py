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
from data.coco_dataset import get_coco_dataloader
from data.voc_dataset import get_voc_dataloader, VOC_CLASSES

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
    # Create model - COCO tiny dataset has 5 classes + background
    model = get_faster_rcnn_model(
        num_classes=6,
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

def load_classes(dataset):
    """
    Load class names from dataset
    
    Args:
        dataset: Dataset type (coco or voc)
        
    Returns:
        class_names: Dictionary of class names
    """
    if dataset == "voc":
        # Pascal VOC has 20 classes
        return {i+1: name for i, name in enumerate(VOC_CLASSES)}
    else:
        # For COCO, we'll create a dummy dataloader to get classes
        dummy_dataloader = get_coco_dataloader(
            root_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coco", "tiny_subset", "val2017"),
            ann_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "coco", "tiny_subset", "annotations", "instances_val2017.json"),
            batch_size=1,
            train=False,
            subset=True
        )
        return dummy_dataloader.dataset.categories

def prepare_image(image_path, device):
    """
    Prepare image for inference
    
    Args:
        image_path: Path to image
        device: Device to run inference on
        
    Returns:
        image_tensor: Tensor for model input
        original_image: Original PIL image
    """
    # Load image
    original_image = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(original_image)
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    return image_tensor, original_image

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    
    Args:
        boxes: Bounding boxes
        scores: Confidence scores
        iou_threshold: IoU threshold for NMS
        
    Returns:
        keep: Indices of boxes to keep
    """
    # Sort boxes by score
    _, sorted_idx = torch.sort(scores, descending=True)
    
    keep = []
    while sorted_idx.size(0) > 0:
        # Keep the box with highest score
        keep.append(sorted_idx[0].item())
        
        # If only one box left, we're done
        if sorted_idx.size(0) == 1:
            break
            
        # Get IoU of the current box with all remaining boxes
        ious = box_iou(boxes[sorted_idx[0]:sorted_idx[0]+1], boxes[sorted_idx[1:]])
        
        # Keep boxes with IoU less than threshold
        mask = ious[0] < iou_threshold
        sorted_idx = sorted_idx[1:][mask]
    
    return keep

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        box1: (N, 4) tensor of boxes [x1, y1, x2, y2]
        box2: (M, 4) tensor of boxes [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) tensor of IoU values
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Compute overlap areas
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / union
    return iou

def detect_objects(model, image_tensor, confidence_threshold=0.5, nms_threshold=0.5):
    """
    Detect objects in an image
    
    Args:
        model: Detection model
        image_tensor: Input image tensor
        confidence_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for NMS
        
    Returns:
        boxes: Detected bounding boxes
        labels: Detected class labels
        scores: Confidence scores
    """
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Get predictions for first image (batch size is 1)
    pred = predictions[0]
    
    # Get boxes, scores, and labels
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]
    
    # Filter by confidence
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Apply NMS
    keep = apply_nms(boxes, scores, nms_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return boxes.cpu(), labels.cpu(), scores.cpu()

def draw_boxes(image, boxes, labels, scores, class_names, output_path):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL image
        boxes: Detected bounding boxes
        labels: Detected class labels
        scores: Confidence scores
        class_names: Dictionary of class names
        output_path: Path to save output image
    """
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = image_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # Generate random colors for each class
    np.random.seed(42)  # For consistent colors
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(1, 100)}
    
    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name and color
        class_id = label.item()
        class_name = class_names.get(class_id, f"Class {class_id}")
        color = colors[class_id]
        
        # Draw rectangle
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        cv2.putText(image_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save image
    cv2.imwrite(output_path, image_cv)
    
    return image_cv

def run_detection(model, image_path, output_dir, class_names, conf_threshold=0.5, nms_threshold=0.5, device="cuda"):
    """
    Run detection on a single image
    
    Args:
        model: Detection model
        image_path: Path to input image
        output_dir: Directory to save output
        class_names: Dictionary of class names
        conf_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for NMS
        device: Device to run inference on
        
    Returns:
        detections: Dictionary with detection results
    """
    # Prepare image
    image_tensor, original_image = prepare_image(image_path, device)
    
    # Detect objects
    boxes, labels, scores = detect_objects(model, image_tensor, conf_threshold, nms_threshold)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get output path
    image_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{image_name}_detection.jpg")
    
    # Draw boxes on image
    annotated_image = draw_boxes(original_image, boxes, labels, scores, class_names, output_path)
    
    # Save detection results
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        class_id = label.item()
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        detection = {
            "box": box.tolist(),
            "label": class_id,
            "class": class_name,
            "score": score.item()
        }
        
        detections.append(detection)
    
    print(f"Detected {len(detections)} objects in {image_path}")
    print(f"Results saved to {output_path}")
    
    return detections

def process_inputs(inputs, output_dir, model, class_names, conf_threshold, nms_threshold, device):
    """
    Process input images or directories
    
    Args:
        inputs: List of input paths (images or directories)
        output_dir: Directory to save outputs
        model: Detection model
        class_names: Dictionary of class names
        conf_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for NMS
        device: Device to run inference on
        
    Returns:
        results: Dictionary with detection results
    """
    # Collect all image paths
    image_paths = []
    
    for input_path in inputs:
        if os.path.isfile(input_path):
            # Single image
            image_paths.append(input_path)
        elif os.path.isdir(input_path):
            # Directory of images
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(glob.glob(os.path.join(input_path, ext)))
    
    # Remove duplicates
    image_paths = list(set(image_paths))
    
    # Check if any images were found
    if not image_paths:
        print("No images found in the provided inputs.")
        return {}
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image
    results = {}
    start_time = time.time()
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        detections = run_detection(model, image_path, output_dir, class_names, conf_threshold, nms_threshold, device)
        results[image_path] = detections
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / len(image_paths)
    
    print(f"Processed {len(image_paths)} images in {elapsed_time:.2f} seconds ({avg_time:.2f} seconds per image)")
    
    return results

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
    class_names = load_classes(args.dataset)
    
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
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="coco",
                      choices=["coco", "voc"],
                      help="Dataset type (for class names)")
    
    args = parser.parse_args()
    
    main(args) 