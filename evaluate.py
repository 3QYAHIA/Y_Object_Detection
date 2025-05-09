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

# Import project modules
from data.coco_dataset import get_coco_dataloader
from models.detector import get_faster_rcnn_model
from utils.visualization import save_detection_visualization

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1: Box in format [x1, y1, x2, y2]
        box2: Box in format [x1, y1, x2, y2]
    
    Returns:
        iou: IoU value
    """
    # Get intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_map(predictions, dataset, iou_thresholds=[0.5, 0.75], conf_threshold=0.5):
    """
    Calculate mAP at different IoU thresholds
    
    Args:
        predictions: List of model predictions (boxes, labels, scores, image_id)
        dataset: Dataset with ground truth
        iou_thresholds: List of IoU thresholds to calculate mAP at
        conf_threshold: Confidence threshold for predictions
    
    Returns:
        results: Dictionary with mAP values at different IoU thresholds
    """
    # Initialize results
    results = {
        'mAP': 0,
        'mAP_0.5': 0,
        'mAP_0.75': 0
    }
    
    # Get all class IDs
    class_ids = set()
    for _, labels, _, _ in predictions:
        class_ids.update(labels.unique().tolist())
    
    # Calculate AP for each class at each IoU threshold
    aps_per_threshold = defaultdict(list)
    
    for class_id in class_ids:
        for iou_threshold in iou_thresholds:
            # Get all predictions for this class
            all_predictions = []
            
            for boxes, labels, scores, image_id in predictions:
                # Filter by class and confidence
                mask = (labels == class_id) & (scores >= conf_threshold)
                
                # Get predictions for this class
                class_boxes = boxes[mask]
                class_scores = scores[mask]
                
                # Add to all predictions
                for box, score in zip(class_boxes, class_scores):
                    all_predictions.append({
                        'image_id': image_id.item(),
                        'box': box.tolist(),
                        'score': score.item()
                    })
            
            # Sort predictions by score (descending)
            all_predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Get all ground truth for this class
            gt_by_image = {}
            
            for i, img_id in enumerate(dataset.ids):
                # Get annotations for this image
                ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
                anns = dataset.coco.loadAnns(ann_ids)
                
                # Get ground truth boxes for this class
                gt_boxes = []
                
                for ann in anns:
                    if dataset.cat_ids_to_continuous.get(ann['category_id'], -1) == class_id:
                        # Get box in [x1, y1, x2, y2] format
                        bbox = ann['bbox']
                        x1, y1, w, h = bbox
                        box = [x1, y1, x1 + w, y1 + h]
                        gt_boxes.append(box)
                
                if gt_boxes:
                    gt_by_image[img_id] = {
                        'boxes': gt_boxes,
                        'matched': [False] * len(gt_boxes)
                    }
            
            # Calculate precision-recall curve
            tp = []
            fp = []
            
            for pred in all_predictions:
                if pred['image_id'] not in gt_by_image:
                    # No ground truth for this image
                    fp.append(1)
                    tp.append(0)
                    continue
                
                # Get ground truth for this image
                gt = gt_by_image[pred['image_id']]
                
                # Find highest IoU match
                max_iou = -1
                max_idx = -1
                
                for i, box in enumerate(gt['boxes']):
                    if not gt['matched'][i]:
                        iou = calculate_iou(pred['box'], box)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = i
                
                if max_iou >= iou_threshold:
                    # Match found
                    gt['matched'][max_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    # No match
                    tp.append(0)
                    fp.append(1)
            
            # Calculate precision-recall curve
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Count total number of ground truth
            total_gt = sum(len(gt['boxes']) for gt in gt_by_image.values())
            
            # Calculate precision and recall
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recall = tp_cumsum / (total_gt + 1e-10)
            
            # Add (0, 1) point to precision-recall curve
            precision = np.concatenate(([1], precision))
            recall = np.concatenate(([0], recall))
            
            # Calculate area under PR curve (AP)
            ap = 0
            for i in range(len(precision) - 1):
                ap += (recall[i + 1] - recall[i]) * precision[i + 1]
            
            # Add AP to results
            aps_per_threshold[iou_threshold].append(ap)
    
    # Calculate mAP for each IoU threshold
    for iou_threshold in iou_thresholds:
        aps = aps_per_threshold[iou_threshold]
        if aps:
            mAP = np.mean(aps)
            if iou_threshold == 0.5:
                results['mAP_0.5'] = mAP
            elif iou_threshold == 0.75:
                results['mAP_0.75'] = mAP
    
    # Calculate average mAP across all IoU thresholds
    all_aps = []
    for aps in aps_per_threshold.values():
        all_aps.extend(aps)
    
    if all_aps:
        results['mAP'] = np.mean(all_aps)
    
    return results

def calculate_precision_recall_f1(predictions, dataset, conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate precision, recall, and F1-score per class
    
    Args:
        predictions: List of model predictions (boxes, labels, scores, image_id)
        dataset: Dataset with ground truth
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to ground truth
    
    Returns:
        metrics_df: DataFrame with precision, recall, and F1-score per class
    """
    # Get all classes
    classes = dataset.categories.copy()
    
    # Create mapping from continuous index to category ID
    idx_to_cat_id = {v: k for k, v in dataset.cat_ids_to_continuous.items()}
    
    # Initialize metrics
    metrics = []
    
    # Process each class
    for class_idx, class_id in idx_to_cat_id.items():
        if class_id not in classes:
            continue
        
        class_name = classes[class_id]
        
        # Count true positives, false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0
        
        # Process each image
        for i, img_id in enumerate(dataset.ids):
            # Get ground truth for this image
            ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
            anns = dataset.coco.loadAnns(ann_ids)
            
            # Get ground truth boxes for this class
            gt_boxes = []
            
            for ann in anns:
                if dataset.cat_ids_to_continuous.get(ann['category_id'], -1) == class_idx:
                    # Get box in [x1, y1, x2, y2] format
                    bbox = ann['bbox']
                    x1, y1, w, h = bbox
                    box = [x1, y1, x1 + w, y1 + h]
                    gt_boxes.append(box)
            
            # Find prediction for this image
            pred_boxes = []
            pred_scores = []
            
            for boxes, labels, scores, image_id in predictions:
                if image_id.item() == img_id:
                    # Filter by class and confidence
                    mask = (labels == class_idx) & (scores >= conf_threshold)
                    
                    # Get predictions for this class
                    pred_boxes = boxes[mask].tolist()
                    pred_scores = scores[mask].tolist()
                    break
            
            # Match predictions to ground truth
            matched_gt = [False] * len(gt_boxes)
            
            for box, score in zip(pred_boxes, pred_scores):
                # Find highest IoU match
                max_iou = -1
                max_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if not matched_gt[i]:
                        iou = calculate_iou(box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = i
                
                if max_iou >= iou_threshold:
                    # Match found
                    matched_gt[max_idx] = True
                    tp += 1
                else:
                    # No match
                    fp += 1
            
            # Count false negatives
            fn += sum(1 for matched in matched_gt if not matched)
        
        # Calculate metrics
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Add to metrics
        metrics.append({
            'class_id': class_id,
            'category': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': tp + fn
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def calculate_confusion_matrix_data(predictions, dataset, conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate confusion matrix
    
    Args:
        predictions: List of model predictions (boxes, labels, scores, image_id)
        dataset: Dataset with ground truth
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to ground truth
    
    Returns:
        cm: Confusion matrix
        class_names: List of class names
    """
    # Get all class indices
    class_indices = list(sorted(set(dataset.cat_ids_to_continuous.values())))
    
    # Get class names
    idx_to_cat_id = {v: k for k, v in dataset.cat_ids_to_continuous.items()}
    class_names = [dataset.categories.get(idx_to_cat_id.get(i, -1), 'unknown') for i in class_indices]
    
    # Initialize confusion matrix
    n_classes = len(class_indices)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Process each image
    for img_id in tqdm(dataset.ids, desc="Calculating confusion matrix"):
        # Get ground truth for this image
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        
        # Get ground truth boxes and labels
        gt_boxes = []
        gt_labels = []
        
        for ann in anns:
            class_idx = dataset.cat_ids_to_continuous.get(ann['category_id'], -1)
            if class_idx in class_indices:
                # Get box in [x1, y1, x2, y2] format
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                box = [x1, y1, x1 + w, y1 + h]
                gt_boxes.append(box)
                gt_labels.append(class_indices.index(class_idx))
        
        # Find prediction for this image
        for boxes, labels, scores, image_id in predictions:
            if image_id.item() == img_id:
                # Filter by confidence
                mask = scores >= conf_threshold
                pred_boxes = boxes[mask].tolist()
                pred_labels = labels[mask].tolist()
                
                # Convert continuous indices to confusion matrix indices
                pred_labels = [class_indices.index(l) if l in class_indices else -1 for l in pred_labels]
                
                # Match predictions to ground truth
                matched_gt = [False] * len(gt_boxes)
                
                for box, pred_label in zip(pred_boxes, pred_labels):
                    if pred_label == -1:
                        continue
                    
                    # Find highest IoU match
                    max_iou = -1
                    max_idx = -1
                    
                    for i, gt_box in enumerate(gt_boxes):
                        if not matched_gt[i]:
                            iou = calculate_iou(box, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                max_idx = i
                    
                    if max_iou >= iou_threshold:
                        # Match found
                        matched_gt[max_idx] = True
                        gt_label = gt_labels[max_idx]
                        cm[gt_label, pred_label] += 1
                
                # Add false negatives
                for i, matched in enumerate(matched_gt):
                    if not matched:
                        gt_label = gt_labels[i]
                        cm[gt_label, gt_label] += 0  # No increment, just to keep track
                
                break
    
    return cm, class_names

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_inference(model, data_loader, device):
    """
    Run inference on a dataset
    
    Args:
        model: Detection model
        data_loader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        predictions: List of model predictions (boxes, labels, scores, image_id)
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Running inference"):
            # Move to device
            images = list(image.to(device) for image in images)
            
            # Run inference
            outputs = model(images)
            
            # Process outputs
            for i, output in enumerate(outputs):
                # Get predictions
                boxes = output['boxes'].cpu()
                labels = output['labels'].cpu()
                scores = output['scores'].cpu()
                image_id = targets[i]['image_id']
                
                # Add to predictions
                predictions.append((boxes, labels, scores, image_id))
    
    return predictions

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
    
    # Create dataloader for evaluation
    val_dataloader = get_coco_dataloader(
        root_dir=root_dir,
        ann_file=ann_file,
        batch_size=args.batch_size,
        train=False,
        subset=(args.dataset_type == "mini")  # Use subset only for mini dataset
    )
    
    # Create model
    num_classes = len(val_dataloader.dataset.categories) + 1  # +1 for background
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
    
    # Start evaluation
    print("Starting evaluation")
    
    # Run inference
    predictions = run_inference(model, val_dataloader, device)
    
    # Calculate mAP
    print("Calculating mAP...")
    map_results = calculate_map(predictions, val_dataloader.dataset)
    
    # Print and save mAP results
    print("mAP Results:")
    print(f"  mAP: {map_results['mAP']:.4f}")
    print(f"  mAP@0.5: {map_results['mAP_0.5']:.4f}")
    print(f"  mAP@0.75: {map_results['mAP_0.75']:.4f}")
    
    with open(os.path.join(output_dir, "map_results.json"), "w") as f:
        json.dump(map_results, f, indent=2)
    
    # Calculate precision, recall, and F1-score
    print("Calculating precision, recall, and F1-score...")
    metrics_df = calculate_precision_recall_f1(predictions, val_dataloader.dataset)
    
    # Print and save precision, recall, and F1-score
    print("\nPrecision, Recall, F1-score:")
    print(metrics_df.to_string(index=False))
    
    metrics_df.to_csv(os.path.join(output_dir, "precision_recall_f1.csv"), index=False)
    
    # Plot precision, recall, and F1-score
    plt.figure(figsize=(12, 6))
    metrics_df.plot(x='category', y=['precision', 'recall', 'f1_score'], kind='bar')
    plt.title('Precision, Recall, and F1-score per Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_f1.png"))
    plt.close()
    
    # Calculate confusion matrix
    print("Calculating confusion matrix...")
    cm, class_names = calculate_confusion_matrix_data(predictions, val_dataloader.dataset)
    
    # Save confusion matrix
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
    with open(os.path.join(output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save example detections
    print("Saving detection visualizations...")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get a batch of images
    images, targets = next(iter(val_dataloader))
    
    # Save visualizations
    save_detection_visualization(
        model=model,
        dataset=val_dataloader.dataset,
        images=images,
        targets=targets,
        output_dir=vis_dir,
        num_samples=min(args.num_vis_samples, len(images))
    )
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                       help="Path to model checkpoint")
    
    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--dataset-type", type=str, default="small",
                      choices=["mini", "small", "full"],
                      help="Type of dataset to use (mini: ~300 images, small: ~5K images, full: ~120K images)")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--num-vis-samples", type=int, default=10,
                       help="Number of samples to visualize")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 