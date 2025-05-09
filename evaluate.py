#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Import project modules
from data.coco_dataset import get_coco_dataloader
from models.detector import get_faster_rcnn_model, get_model_info
from utils.visualization import save_detection_visualization
from evaluation.metrics import (
    calculate_mAP,
    calculate_precision_recall_f1,
    calculate_confusion_matrix
)

def evaluate_model_speed(model, device, image_size=(512, 512), num_iterations=100):
    """
    Evaluate model inference speed
    
    Args:
        model: Detection model
        device: Device to evaluate on
        image_size: Size of input images (height, width)
        num_iterations: Number of iterations to run
        
    Returns:
        fps: Average frames per second
        latency: Average latency in milliseconds
    """
    # Create dummy input - Faster R-CNN expects a list of tensors
    dummy_input = [torch.randn(3, *image_size).to(device)]
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device != torch.device("cpu"):
        torch.cuda.synchronize()
    
    # Measure time
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device != torch.device("cpu"):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time
    latency = (elapsed_time / num_iterations) * 1000  # ms
    
    return fps, latency

def main(args):
    # Set up output directory
    output_dir = os.path.join("outputs", args.backbone, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data root
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Set up data paths - prefer tiny_subset
    tiny_subset_dir = os.path.join(data_root, "coco", "tiny_subset")
    if os.path.exists(tiny_subset_dir) or args.tiny:
        root_dir = os.path.join(data_root, "coco", "tiny_subset", "val2017")
        ann_file = os.path.join(data_root, "coco", "tiny_subset", "annotations", "instances_val2017.json")
        print("Using tiny subset of COCO dataset (5 classes, ~300 images)")
    elif args.subset:
        root_dir = os.path.join(data_root, "coco", "subset", "train2017")
        ann_file = os.path.join(data_root, "coco", "subset", "annotations", "instances_train2017.json")
        print("Using subset of COCO dataset (10 classes, ~1000 images)")
    else:
        root_dir = os.path.join(data_root, "coco", "val2017")
        ann_file = os.path.join(data_root, "coco", "annotations", "instances_val2017.json")
        print("Using full validation set of COCO dataset")
    
    # Create dataloader for evaluation
    val_dataloader = get_coco_dataloader(
        root_dir=root_dir,
        ann_file=ann_file,
        batch_size=args.batch_size,
        train=False,
        subset=args.subset or args.tiny
    )
    
    # Create model
    num_classes = len(val_dataloader.dataset.categories) + 1  # +1 for background
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=False,
        trainable_backbone_layers=0
    )
    
    # Print model info
    model_info = get_model_info(model)
    print("Model Info:")
    print(f"  Backbone: {model_info['backbone']}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    
    # Load model weights
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find checkpoint in default location
        default_path = os.path.join("outputs", args.backbone, "best_model.pth")
        checkpoint_path = default_path if os.path.exists(default_path) else None
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle both full checkpoint dictionary and just state_dict
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
                
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with initialized model...")
    else:
        print("No checkpoint found, using initialized model")
    
    # Move model to device
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Start evaluation
    print("Starting evaluation")
    
    # Evaluate model speed
    fps, latency = evaluate_model_speed(model, device)
    print(f"Model Speed:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency:.2f} ms")
    
    # Save speed metrics
    speed_metrics = {
        'fps': fps,
        'latency': latency
    }
    with open(os.path.join(output_dir, "speed_metrics.json"), "w") as f:
        json.dump(speed_metrics, f, indent=4)
    
    # Run evaluation on test images
    print("Running model inference...")
    predictions = []
    
    for images, targets in tqdm(val_dataloader):
        # Move to device
        images = list(image.to(device) for image in images)
        
        # Run inference
        with torch.no_grad():
            outputs = model(images)
        
        # Collect predictions
        for i, output in enumerate(outputs):
            # Get image id
            image_id = targets[i]['image_id']
            
            # Add to predictions
            predictions.append((
                output['boxes'].cpu(),
                output['labels'].cpu(),
                output['scores'].cpu(),
                image_id.cpu()
            ))
            
            # Only process the requested number of test images
            if len(predictions) >= args.num_test_images:
                break
        
        # Break if we've processed enough images
        if len(predictions) >= args.num_test_images:
            break
    
    print(f"Processed {len(predictions)} test images")
    
    # Save example detections
    print("Saving detection visualizations...")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    # Get a batch of images
    batch_imgs, batch_targets = next(iter(val_dataloader))
    
    # Save visualizations
    save_detection_visualization(
        model=model,
        dataset=val_dataloader.dataset,
        images=batch_imgs,
        targets=batch_targets,
        output_dir=vis_dir,
        num_samples=min(args.num_vis_samples, len(batch_imgs)),
        threshold=args.conf_threshold
    )
    
    # Calculate mAP
    print("Calculating mAP...")
    mAP_results = calculate_mAP(
        dataset=val_dataloader.dataset,
        predictions=predictions,
        output_dir=output_dir
    )
    
    # Print mAP results
    print("mAP Results:")
    if mAP_results and 'mAP' in mAP_results:
        print(f"  mAP: {mAP_results['mAP']:.4f}")
        print(f"  mAP@0.5: {mAP_results['mAP_0.5']:.4f}")
        print(f"  mAP@0.75: {mAP_results['mAP_0.75']:.4f}")
    else:
        print("  No valid detections for mAP calculation")
    
    # Calculate precision, recall, F1-score
    print("Calculating precision, recall, F1-score...")
    try:
        pr_results = calculate_precision_recall_f1(
            dataset=val_dataloader.dataset,
            predictions=predictions,
            iou_threshold=0.5,
            conf_threshold=args.conf_threshold,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error calculating precision/recall metrics: {e}")
        pr_results = None
    
    # Calculate confusion matrix
    print("Calculating confusion matrix...")
    try:
        cm = calculate_confusion_matrix(
            dataset=val_dataloader.dataset,
            predictions=predictions,
            conf_threshold=args.conf_threshold,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error calculating confusion matrix: {e}")
        cm = None
    
    print("Evaluation complete. Results saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                       help="Path to model checkpoint")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--num-test-images", type=int, default=100,
                       help="Number of test images to evaluate")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--subset", action="store_true",
                       help="Use subset of COCO dataset (10 classes)")
    parser.add_argument("--tiny", action="store_true", default=True,
                       help="Use tiny subset of COCO dataset (5 classes, <300MB)")
    
    # Output parameters
    parser.add_argument("--num-vis-samples", type=int, default=10,
                       help="Number of samples to visualize")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 