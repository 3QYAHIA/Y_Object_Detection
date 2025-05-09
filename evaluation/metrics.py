import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import pandas as pd
import seaborn as sns
from collections import defaultdict

def convert_to_coco_format(dataset, predictions, output_file=None):
    """
    Convert model predictions to COCO format for evaluation
    
    Args:
        dataset: Dataset with category information
        predictions: List of model predictions
        output_file: Optional file to save predictions
        
    Returns:
        coco_dt: COCO detection results
    """
    # Create COCO results
    coco_results = []
    
    # Process predictions
    for i, (pred_boxes, pred_labels, pred_scores, img_id) in enumerate(predictions):
        # Convert to numpy arrays
        boxes = pred_boxes.cpu().numpy()
        labels = pred_labels.cpu().numpy()
        scores = pred_scores.cpu().numpy()
        
        # Process each box
        for box_id in range(len(boxes)):
            x1, y1, x2, y2 = boxes[box_id]
            label = labels[box_id]
            score = scores[box_id]
            
            # Convert box to COCO format [x, y, width, height]
            width = x2 - x1
            height = y2 - y1
            
            # Create detection entry
            result = {
                'image_id': img_id.item(),
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(score)
            }
            
            coco_results.append(result)
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(coco_results, f)
    
    # Create COCO detection object
    coco_dt = dataset.coco.loadRes(coco_results) if coco_results else None
    
    return coco_dt

def calculate_mAP(dataset, predictions, output_dir=None):
    """
    Calculate mAP using COCO evaluation tools
    
    Args:
        dataset: Dataset with COCO annotations
        predictions: List of model predictions
        output_dir: Optional directory to save results
        
    Returns:
        results: Dictionary of evaluation results
    """
    # Convert predictions to COCO format
    coco_gt = dataset.coco
    coco_dt = convert_to_coco_format(dataset, predictions, 
                                     output_file=os.path.join(output_dir, 'predictions.json') if output_dir else None)
    
    if coco_dt is None:
        print("No valid detections for evaluation")
        return {}
    
    # Create COCO evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract results
    results = {
        'mAP_0.5': coco_eval.stats[1],  # AP at IoU=0.5
        'mAP_0.75': coco_eval.stats[2],  # AP at IoU=0.75
        'mAP': coco_eval.stats[0],       # AP at IoU=0.5:0.95
        'mAP_small': coco_eval.stats[3],  # AP for small objects
        'mAP_medium': coco_eval.stats[4], # AP for medium objects
        'mAP_large': coco_eval.stats[5],  # AP for large objects
    }
    
    # Save results to file if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'mAP_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create a summary table
        summary = pd.DataFrame([results])
        summary.to_csv(os.path.join(output_dir, 'mAP_summary.csv'), index=False)
    
    return results

def calculate_precision_recall_f1(dataset, predictions, iou_threshold=0.5, conf_threshold=0.5, output_dir=None):
    """
    Calculate precision, recall and F1-score per class
    
    Args:
        dataset: Dataset with category information
        predictions: List of model predictions
        iou_threshold: IoU threshold for matching predictions to ground truth
        conf_threshold: Confidence threshold for counting predictions
        output_dir: Optional directory to save results
        
    Returns:
        metrics: Dictionary of metrics per class
    """
    # Create dictionaries to store metrics
    class_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    # Process each image
    for i, (pred_boxes, pred_labels, pred_scores, img_id) in enumerate(predictions):
        # Get ground truth for this image
        gt_ann_ids = dataset.coco.getAnnIds(imgIds=img_id.item())
        gt_anns = dataset.coco.loadAnns(gt_ann_ids)
        
        # Create ground truth dictionary per class
        gt_per_class = defaultdict(list)
        for ann in gt_anns:
            cat_id = ann['category_id']
            cont_id = dataset.cat_ids_to_continuous[cat_id]
            x, y, w, h = ann['bbox']
            gt_per_class[cont_id].append([x, y, x + w, y + h])
        
        # Convert predictions to numpy
        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        # Filter by confidence
        mask = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        # Create prediction dictionary per class
        pred_per_class = defaultdict(list)
        for j in range(len(pred_boxes)):
            label = pred_labels[j]
            box = pred_boxes[j]
            pred_per_class[label].append((box, j))
        
        # For each class, calculate IoU between predictions and ground truth
        for label in set(list(gt_per_class.keys()) + list(pred_per_class.keys())):
            gt_boxes = np.array(gt_per_class[label])
            pred_boxes_with_idx = pred_per_class[label]
            
            # Count false negatives if no predictions for this class
            if len(pred_boxes_with_idx) == 0:
                class_metrics[label]['FN'] += len(gt_boxes)
                continue
                
            # Count false positives if no ground truth for this class
            if len(gt_boxes) == 0:
                class_metrics[label]['FP'] += len(pred_boxes_with_idx)
                continue
            
            # Extract boxes and indices
            pred_boxes_class = np.array([box for box, idx in pred_boxes_with_idx])
            pred_indices = [idx for box, idx in pred_boxes_with_idx]
            
            # Calculate IoU for all combinations of gt and pred boxes
            ious = calculate_iou_matrix(gt_boxes, pred_boxes_class)
            
            # Match predictions to ground truth
            matched_gt = set()
            matched_pred = set()
            
            # Sort IoUs in descending order
            iou_indices = np.dstack(np.unravel_index(np.argsort(ious.ravel())[::-1], ious.shape))[0]
            
            # Match predictions to ground truth greedily
            for gt_idx, pred_idx in iou_indices:
                if gt_idx in matched_gt or pred_idx in matched_pred:
                    continue
                    
                if ious[gt_idx, pred_idx] >= iou_threshold:
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    class_metrics[label]['TP'] += 1
            
            # Count false positives (predictions without matching ground truth)
            class_metrics[label]['FP'] += len(pred_boxes_with_idx) - len(matched_pred)
            
            # Count false negatives (ground truth without matching prediction)
            class_metrics[label]['FN'] += len(gt_boxes) - len(matched_gt)
    
    # Calculate precision, recall and F1-score for each class
    metrics = {}
    for label, counts in class_metrics.items():
        # Skip background class
        if label == 0:
            continue
            
        TP = counts['TP']
        FP = counts['FP']
        FN = counts['FN']
        
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # Get category name
        cat_name = None
        for cat_id, cont_id in dataset.cat_ids_to_continuous.items():
            if cont_id == label:
                cat_name = dataset.categories[cat_id]
                break
        
        metrics[label] = {
            'category': cat_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }
    
    # Save results to file if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame from metrics
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.index.name = 'class_id'
        df.reset_index(inplace=True)
        
        # Save metrics to CSV
        df.to_csv(os.path.join(output_dir, 'precision_recall_f1.csv'), index=False)
        
        # Plot precision, recall and F1-score
        plt.figure(figsize=(12, 6))
        bar_width = 0.2
        index = np.arange(len(metrics))
        
        plt.bar(index, df['precision'], bar_width, label='Precision')
        plt.bar(index + bar_width, df['recall'], bar_width, label='Recall')
        plt.bar(index + 2 * bar_width, df['f1_score'], bar_width, label='F1-score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Precision, Recall and F1-score per class')
        plt.xticks(index + bar_width, df['category'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'))
        plt.close()
    
    return metrics

def calculate_confusion_matrix(dataset, predictions, conf_threshold=0.5, output_dir=None):
    """
    Calculate confusion matrix for class predictions
    
    Args:
        dataset: Dataset with category information
        predictions: List of model predictions
        conf_threshold: Confidence threshold for counting predictions
        output_dir: Optional directory to save results
        
    Returns:
        cm: Confusion matrix
    """
    y_true = []
    y_pred = []
    
    # Process each image
    for i, (pred_boxes, pred_labels, pred_scores, img_id) in enumerate(predictions):
        # Get ground truth for this image
        gt_ann_ids = dataset.coco.getAnnIds(imgIds=img_id.item())
        gt_anns = dataset.coco.loadAnns(gt_ann_ids)
        
        # Collect ground truth labels
        for ann in gt_anns:
            cat_id = ann['category_id']
            cont_id = dataset.cat_ids_to_continuous[cat_id]
            y_true.append(cont_id)
        
        # Collect predicted labels
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        # Filter by confidence
        mask = pred_scores >= conf_threshold
        pred_labels = pred_labels[mask]
        
        # Add predicted labels
        for label in pred_labels:
            y_pred.append(label)
    
    # Get all classes
    classes = sorted(list(set(y_true + y_pred)))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Save results to file if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get category names
        class_names = []
        for class_id in classes:
            cat_name = None
            for cat_id, cont_id in dataset.cat_ids_to_continuous.items():
                if cont_id == class_id:
                    cat_name = dataset.categories[cat_id]
                    break
            class_names.append(cat_name)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save confusion matrix to CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    
    return cm

def calculate_iou_matrix(boxes_a, boxes_b):
    """
    Calculate IoU between all pairs of boxes in boxes_a and boxes_b
    
    Args:
        boxes_a: Array of shape (N, 4) in [x1, y1, x2, y2] format
        boxes_b: Array of shape (M, 4) in [x1, y1, x2, y2] format
        
    Returns:
        ious: Matrix of shape (N, M) containing IoU values
    """
    # If either array is empty, return empty result
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    # Expand dimensions to allow broadcasting
    boxes_a = boxes_a[:, np.newaxis, :]  # (N, 1, 4)
    boxes_b = boxes_b[np.newaxis, :, :]  # (1, M, 4)
    
    # Calculate intersection
    max_xy = np.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    min_xy = np.maximum(boxes_a[..., :2], boxes_b[..., :2])
    
    # Calculate intersection area
    intersection = np.maximum(0, max_xy - min_xy)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    
    # Calculate areas of boxes
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    
    # Calculate union area
    union_area = area_a + area_b - intersection_area
    
    # Calculate IoU
    iou = np.where(union_area > 0, intersection_area / union_area, 0)
    
    return iou 