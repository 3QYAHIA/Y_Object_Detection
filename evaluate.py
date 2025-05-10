#!/usr/bin/env python3
# Set matplotlib backend to non-interactive to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import torch
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Import project modules
from data.coco_dataset import get_coco_dataloader
from data.voc_dataset import get_voc_dataloader
from models.detector import get_faster_rcnn_model
from utils.visualization import save_detection_visualization

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        iou: IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

def calculate_map(precision_list, recall_list, classes, iou_thresholds=[0.5, 0.75]):
    """
    Calculate mean Average Precision (mAP) at different IoU thresholds
    
    Args:
        precision_list: List of precision values per class at different IoU thresholds
        recall_list: List of recall values per class at different IoU thresholds
        classes: Class names
        iou_thresholds: IoU thresholds to calculate mAP at
        
    Returns:
        map_values: Dictionary with mAP values at different IoU thresholds
    """
    mAP = {}
    
    # Calculate mAP for each IoU threshold
    for i, iou_threshold in enumerate(iou_thresholds):
        if i >= len(precision_list):
            continue
            
        # Calculate AP for each class
        ap_per_class = []
        for c in range(len(classes)):
            if c >= len(precision_list[i]):
                continue
                
            # Get precision and recall values for this class
            precision = precision_list[i][c]
            recall = recall_list[i][c]
            
            # Skip if no detections
            if len(precision) == 0:
                ap_per_class.append(0.0)
                continue
            
            # Calculate average precision using 11-point interpolation
            ap = 0.0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.0
            
            ap_per_class.append(ap)
        
        # Calculate mAP
        mAP[f"mAP@{iou_threshold}"] = np.mean(ap_per_class)
    
    return mAP

def calculate_metrics(model, data_loader, device, num_classes, dataset_type="coco", output_dir=None):
    """
    Calculate evaluation metrics for object detection model
    
    Args:
        model: Detection model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
        num_classes: Number of classes
        dataset_type: Type of dataset (coco or voc)
        output_dir: Directory to save outputs
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    # Get class names
    class_names = list(data_loader.dataset.categories.values())
    
    # Initialize metrics
    all_ground_truths = []
    all_predictions = []
    
    # Confusion matrix
    conf_matrix = np.zeros((num_classes - 1, num_classes - 1))  # Exclude background class
    
    # Precision and recall at different IoU thresholds
    iou_thresholds = [0.5, 0.75]
    precision_list = [[] for _ in iou_thresholds]
    recall_list = [[] for _ in iou_thresholds]
    
    # Class-wise metrics
    class_metrics = {c: {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0
    } for c in range(1, num_classes)}  # Classes start from 1 (0 is background)
    
    print("Calculating metrics...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Process each image
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # Get ground truth boxes and labels
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                
                # Get predictions with score > 0.5
                scores = output["scores"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()[scores > 0.5]
                pred_labels = output["labels"].cpu().numpy()[scores > 0.5]
                pred_scores = scores[scores > 0.5]
                
                # Skip if no predictions or no ground truths
                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue
                
                # Store ground truths and predictions
                all_ground_truths.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels
                })
                
                all_predictions.append({
                    "boxes": pred_boxes,
                    "labels": pred_labels,
                    "scores": pred_scores
                })
                
                # Calculate IoU between predictions and ground truths
                ious = np.zeros((len(pred_boxes), len(gt_boxes)))
                for j, pred_box in enumerate(pred_boxes):
                    for k, gt_box in enumerate(gt_boxes):
                        ious[j, k] = calculate_iou(pred_box, gt_box)
                
                # Match predictions to ground truths for confusion matrix
                for iou_idx, iou_threshold in enumerate(iou_thresholds):
                    # Initialize precision and recall for this image
                    class_precision = [[] for _ in range(num_classes)]
                    class_recall = [[] for _ in range(num_classes)]
                    
                    # For each prediction, find the best matching ground truth
                    matched_gt = set()
                    
                    # Sort predictions by score
                    sorted_idx = np.argsort(-pred_scores)
                    
                    for j in sorted_idx:
                        # Get class
                        pred_class = pred_labels[j]
                        
                        # Skip background class
                        if pred_class == 0:
                            continue
                        
                        # Get ground truths of this class
                        gt_idx = np.where(gt_labels == pred_class)[0]
                        
                        # Skip if no ground truths of this class
                        if len(gt_idx) == 0:
                            class_precision[pred_class].append(0.0)
                            continue
                        
                        # Find the best matching ground truth
                        best_iou = 0
                        best_gt = -1
                        
                        for k in gt_idx:
                            if k in matched_gt:
                                continue
                                
                            if ious[j, k] > best_iou:
                                best_iou = ious[j, k]
                                best_gt = k
                        
                        # Check if match is good enough
                        if best_iou >= iou_threshold and best_gt != -1:
                            matched_gt.add(best_gt)
                            class_precision[pred_class].append(1.0)
                            
                            # Update confusion matrix (only for IoU 0.5)
                            if iou_idx == 0:  # IoU 0.5
                                conf_matrix[pred_class - 1, gt_labels[best_gt] - 1] += 1
                        else:
                            class_precision[pred_class].append(0.0)
                    
                    # Calculate class recall
                    for c in range(1, num_classes):
                        gt_count = np.sum(gt_labels == c)
                        if gt_count > 0:
                            tp = np.sum([1 for j in matched_gt if gt_labels[j] == c])
                            class_recall[c].extend([1.0] * tp + [0.0] * (gt_count - tp))
                    
                    # Add to global lists
                    for c in range(1, num_classes):
                        precision_list[iou_idx].append(class_precision[c])
                        recall_list[iou_idx].append(class_recall[c])
                
                # Update class metrics (at IoU 0.5)
                for c in range(1, num_classes):
                    gt_count = np.sum(gt_labels == c)
                    pred_count = np.sum(pred_labels == c)
                    
                    if gt_count > 0 or pred_count > 0:
                        # Calculate IoU matrix for this class
                        class_ious = np.zeros((pred_count, gt_count))
                        pred_idx = np.where(pred_labels == c)[0]
                        gt_idx = np.where(gt_labels == c)[0]
                        
                        for j, p_idx in enumerate(pred_idx):
                            for k, g_idx in enumerate(gt_idx):
                                class_ious[j, k] = ious[p_idx, g_idx]
                        
                        # Match predictions to ground truths
                        matched_gt = set()
                        true_positives = 0
                        
                        # Sort predictions by score
                        if pred_count > 0:
                            scores_idx = np.argsort(-pred_scores[pred_labels == c])
                            
                            for j in scores_idx:
                                # Find the best matching ground truth
                                if gt_count > 0:
                                    best_iou = 0
                                    best_gt = -1
                                    
                                    for k in range(gt_count):
                                        if k in matched_gt:
                                            continue
                                            
                                        if class_ious[j, k] > best_iou and class_ious[j, k] >= 0.5:
                                            best_iou = class_ious[j, k]
                                            best_gt = k
                                    
                                    # Check if match is good enough
                                    if best_gt != -1:
                                        matched_gt.add(best_gt)
                                        true_positives += 1
                        
                        # Update metrics
                        false_positives = pred_count - true_positives
                        false_negatives = gt_count - true_positives
                        
                        # Calculate precision, recall, F1
                        if true_positives + false_positives > 0:
                            precision = true_positives / (true_positives + false_positives)
                        else:
                            precision = 0.0
                            
                        if true_positives + false_negatives > 0:
                            recall = true_positives / (true_positives + false_negatives)
                        else:
                            recall = 0.0
                            
                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                        else:
                            f1 = 0.0
                        
                        # Update class metrics
                        class_metrics[c]["precision"] += precision
                        class_metrics[c]["recall"] += recall
                        class_metrics[c]["f1"] += f1
                        class_metrics[c]["support"] += gt_count
    
    # Calculate mean metrics
    for c in range(1, num_classes):
        if class_metrics[c]["support"] > 0:
            class_metrics[c]["precision"] /= len(data_loader)
            class_metrics[c]["recall"] /= len(data_loader)
            class_metrics[c]["f1"] /= len(data_loader)
    
    # Calculate mAP
    mAP = calculate_map(precision_list, recall_list, class_names, iou_thresholds)
    
    # Calculate overall metrics
    overall_metrics = {
        "precision": np.mean([m["precision"] for m in class_metrics.values()]),
        "recall": np.mean([m["recall"] for m in class_metrics.values()]),
        "f1": np.mean([m["f1"] for m in class_metrics.values()]),
    }
    
    # Combine all metrics
    metrics = {
        "overall": overall_metrics,
        "per_class": class_metrics,
        "mAP": mAP,
        "num_images": len(data_loader.dataset)
    }
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Number of images: {metrics['num_images']}")
    print(f"mAP@0.5: {mAP.get('mAP@0.5', 0):.4f}")
    print(f"mAP@0.75: {mAP.get('mAP@0.75', 0):.4f}")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1 Score: {overall_metrics['f1']:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    print(f"{'Class': <15} {'Precision': <10} {'Recall': <10} {'F1 Score': <10} {'Support': <10}")
    print("-" * 60)
    
    for c in range(1, num_classes):
        class_name = class_names[c - 1] if c <= len(class_names) else f"Class {c}"
        print(f"{class_name: <15} {class_metrics[c]['precision']:.4f}      {class_metrics[c]['recall']:.4f}      {class_metrics[c]['f1']:.4f}      {class_metrics[c]['support']}")
    
    # Visualize confusion matrix
    if output_dir is not None:
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)
        
        # Create confusion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt=".0f", 
                   xticklabels=class_names[:num_classes-1], 
                   yticklabels=class_names[:num_classes-1])
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Create precision-recall curve
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes-1))
        
        for c in range(1, num_classes):
            if c <= len(precision_list[0]):
                precision = np.array(precision_list[0][c-1])
                recall = np.array(recall_list[0][c-1])
                
                if len(precision) > 0:
                    # Sort by recall
                    idx = np.argsort(recall)
                    recall = recall[idx]
                    precision = precision[idx]
                    
                    # Plot precision-recall curve
                    class_name = class_names[c - 1] if c <= len(class_names) else f"Class {c}"
                    plt.plot(recall, precision, label=class_name, color=colors[c-1])
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()
    
    return metrics

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    # Set up output directory
    output_dir = os.path.join("evaluation", args.backbone)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data root
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Set up dataloaders based on dataset type
    if args.dataset == "voc":
        # Create dataloaders for Pascal VOC
        dataloader = get_voc_dataloader(
            root_dir=data_root,
            year=args.voc_year,
            image_set=args.voc_val_set,
            batch_size=args.batch_size,
            download=False
        )
        
        # Get number of classes
        num_classes = len(dataloader.dataset.categories) + 1  # +1 for background
        print(f"Using Pascal VOC {args.voc_year} dataset with {num_classes-1} classes")
        dataset_type = "voc"
    
    else:  # Default to COCO dataset
        # Set up data paths based on dataset type
        if args.dataset_type == "mini":
            # Use the tiny subset
            root_dir = os.path.join(data_root, "coco", "tiny_subset", "val2017")
            ann_file = os.path.join(data_root, "coco", "tiny_subset", "annotations", "instances_val2017.json")
            print("Using mini COCO dataset (5 classes, ~300 images)")
        elif args.dataset_type == "small":
            # Use val2017
            root_dir = os.path.join(data_root, "coco", "val2017")
            ann_file = os.path.join(data_root, "coco", "annotations", "instances_val2017.json")
            print("Using COCO val2017 dataset (~5K images)")
        elif args.dataset_type == "full":
            # Use train2017
            root_dir = os.path.join(data_root, "coco", "train2017")
            ann_file = os.path.join(data_root, "coco", "annotations", "instances_train2017.json")
            print("Using COCO train2017 dataset (~120K images)")
        
        # Check if dataset exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory {root_dir} not found. Please check your paths.")
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file {ann_file} not found. Please check your paths.")
        
        # Create dataloader
        dataloader = get_coco_dataloader(
            root_dir=root_dir,
            ann_file=ann_file,
            batch_size=args.batch_size,
            train=False,
            subset=(args.dataset_type == "mini")  # Use subset only for mini dataset
        )
        
        # Get number of classes
        num_classes = len(dataloader.dataset.categories) + 1  # +1 for background
        dataset_type = "coco"
    
    # Load model
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=False
    )
    
    # Load model weights
    model_path = os.path.join("outputs", args.backbone, "best_model.pth")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No saved model found at {model_path}, using untrained model")
    
    # Move model to device
    model.to(device)
    
    # Evaluate model
    start_time = time.time()
    metrics = calculate_metrics(model, dataloader, device, num_classes, dataset_type, output_dir)
    elapsed_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get a batch of data
        imgs, targets = next(iter(dataloader))
        
        # Save visualizations
        save_detection_visualization(
            model=model,
            dataset=dataloader.dataset,
            images=imgs,
            targets=targets,
            output_dir=vis_dir,
            num_samples=min(len(imgs), args.num_vis)
        )
        
        print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    
    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--dataset", type=str, default="coco",
                      choices=["coco", "voc"],
                      help="Dataset to use (coco or pascal voc)")
    parser.add_argument("--dataset-type", type=str, default="small",
                      choices=["mini", "small", "full"],
                      help="Type of COCO dataset (mini: ~300 images, small: ~5K images, full: ~120K images)")
    
    # Pascal VOC specific parameters
    parser.add_argument("--voc-year", type=str, default="2012",
                      choices=["2007", "2008", "2009", "2010", "2011", "2012"],
                      help="Pascal VOC dataset year")
    parser.add_argument("--voc-val-set", type=str, default="val",
                      choices=["val", "test"],
                      help="Pascal VOC validation image set")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Visualization parameters
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Create visualizations")
    parser.add_argument("--num-vis", type=int, default=10,
                       help="Number of visualizations to create")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 