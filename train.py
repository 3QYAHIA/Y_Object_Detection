#!/usr/bin/env python3
# Set matplotlib backend to non-interactive to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import torch
import numpy as np
import argparse
import time
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import requests
from pathlib import Path

# Import project modules
from data.coco_dataset import get_coco_dataloader
from models.detector import get_faster_rcnn_model, get_model_info
from utils.visualization import save_detection_visualization

def download_coco_dataset(data_dir, dataset_type="mini"):
    """
    Download COCO dataset
    
    Args:
        data_dir: Directory to save dataset
        dataset_type: Type of dataset to download (mini, small, full)
            - mini: 5 classes, ~300 images (default)
            - small: COCO val2017 - ~5K images
            - full: COCO train2017 - ~120K images
    """
    os.makedirs(data_dir, exist_ok=True)
    coco_dir = os.path.join(data_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)
    
    if dataset_type == "mini":
        print("Using mini COCO dataset (5 classes, ~300 images)")
        # Mini dataset is already included in the project
        return
    
    elif dataset_type == "small":
        print("Downloading COCO val2017 dataset (~5K images)...")
        # Download val2017 images
        val_url = "http://images.cocodataset.org/zips/val2017.zip"
        val_zip = os.path.join(coco_dir, "val2017.zip")
        
        # Download annotations
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = os.path.join(coco_dir, "annotations.zip")
        
        # Download files
        if not os.path.exists(os.path.join(coco_dir, "val2017")):
            print("Downloading val2017 images...")
            download_file(val_url, val_zip)
            extract_zip(val_zip, coco_dir)
        
        if not os.path.exists(os.path.join(coco_dir, "annotations")):
            print("Downloading annotations...")
            download_file(ann_url, ann_zip)
            extract_zip(ann_zip, coco_dir)
            
    elif dataset_type == "full":
        print("Downloading COCO train2017 dataset (~120K images)...")
        # Download train2017 images
        train_url = "http://images.cocodataset.org/zips/train2017.zip"
        train_zip = os.path.join(coco_dir, "train2017.zip")
        
        # Download annotations
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = os.path.join(coco_dir, "annotations.zip")
        
        # Download files
        if not os.path.exists(os.path.join(coco_dir, "train2017")):
            print("Downloading train2017 images...")
            download_file(train_url, train_zip)
            extract_zip(train_zip, coco_dir)
        
        if not os.path.exists(os.path.join(coco_dir, "annotations")):
            print("Downloading annotations...")
            download_file(ann_url, ann_zip)
            extract_zip(ann_zip, coco_dir)
    
    print("Dataset download complete.")

def download_file(url, local_path):
    """Download a file from a URL to a local path"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path) as pbar:
            for data in response.iter_content(chunk_size=4096):
                f.write(data)
                pbar.update(len(data))

def extract_zip(zip_path, extract_dir):
    """Extract a zip file"""
    import zipfile
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extraction complete.")

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: Detection model
        optimizer: Optimizer
        data_loader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    # Set model to training mode
    model.train()
    
    # Track metrics
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    
    # Iterate over batches
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for NaN loss
        if torch.isnan(losses):
            print("NaN loss detected")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += losses.item()
        
        # Update individual losses
        for loss_name, loss_value in loss_dict.items():
            if loss_name == "loss_classifier":
                loss_classifier += loss_value.item()
            elif loss_name == "loss_box_reg":
                loss_box_reg += loss_value.item()
            elif loss_name == "loss_objectness":
                loss_objectness += loss_value.item()
            elif loss_name == "loss_rpn_box_reg":
                loss_rpn_box_reg += loss_value.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses.item(),
            'average': total_loss / (i + 1)
        })
    
    # Calculate average losses
    num_batches = len(data_loader)
    epoch_loss = total_loss / num_batches
    epoch_loss_classifier = loss_classifier / num_batches
    epoch_loss_box_reg = loss_box_reg / num_batches
    epoch_loss_objectness = loss_objectness / num_batches
    epoch_loss_rpn_box_reg = loss_rpn_box_reg / num_batches
    
    # Log losses
    print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")
    print(f"  Classifier: {epoch_loss_classifier:.4f}")
    print(f"  Box Reg: {epoch_loss_box_reg:.4f}")
    print(f"  Objectness: {epoch_loss_objectness:.4f}")
    print(f"  RPN Box Reg: {epoch_loss_rpn_box_reg:.4f}")
    
    # Return losses
    return {
        'loss': epoch_loss,
        'loss_classifier': epoch_loss_classifier,
        'loss_box_reg': epoch_loss_box_reg,
        'loss_objectness': epoch_loss_objectness,
        'loss_rpn_box_reg': epoch_loss_rpn_box_reg
    }

def evaluate(model, data_loader, device):
    """Simple evaluation function to test on validation data"""
    model.eval()
    total_correct = 0
    total_objects = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Get predictions with confidence > 0.5
                mask = output['scores'] > 0.5
                pred_boxes = output['boxes'][mask].cpu()
                pred_labels = output['labels'][mask].cpu()
                
                # Get ground truth
                gt_boxes = targets[i]['boxes'].to(device)
                gt_labels = targets[i]['labels'].to(device)
                
                # Match predictions to ground truth
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    # Calculate IoU between predicted and ground truth boxes
                    ious = box_iou(pred_boxes, gt_boxes)
                    
                    # For each prediction, find the best matching ground truth
                    max_iou_values, max_iou_idxs = ious.max(dim=1)
                    
                    # Count correct predictions (IoU > 0.5 and correct class)
                    for j, (iou, pred_idx) in enumerate(zip(max_iou_values, max_iou_idxs)):
                        if iou > 0.5 and pred_labels[j] == gt_labels[pred_idx]:
                            total_correct += 1
                
                total_objects += len(gt_boxes)
    
    accuracy = total_correct / max(1, total_objects)
    print(f"Validation accuracy: {accuracy:.4f} ({total_correct}/{total_objects})")
    return accuracy

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: (N, 4) tensor of boxes [x1, y1, x2, y2]
        boxes2: (M, 4) tensor of boxes [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) tensor of IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute overlap areas
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / union
    return iou

def main(args):
    # Set up output directory
    output_dir = os.path.join("outputs", args.backbone)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data root
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Download dataset if needed
    download_coco_dataset(data_root, args.dataset_type)
    
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
        raise FileNotFoundError(f"Dataset directory {root_dir} not found. Please check your paths or use --download to download the dataset.")
    
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file {ann_file} not found. Please check your paths or use --download to download the dataset.")
    
    # Create dataloaders
    train_dataloader = get_coco_dataloader(
        root_dir=root_dir,
        ann_file=ann_file,
        batch_size=args.batch_size,
        train=True,
        subset=(args.dataset_type == "mini")  # Use subset only for mini dataset
    )
    
    # Create a validation dataloader with a different batch size
    val_dataloader = get_coco_dataloader(
        root_dir=root_dir,
        ann_file=ann_file,
        batch_size=args.batch_size // 2 or 1,  # Smaller batch size for validation
        train=False,
        subset=(args.dataset_type == "mini")  # Use subset only for mini dataset
    )
    
    # Create model
    num_classes = len(train_dataloader.dataset.categories) + 1  # +1 for background
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
        trainable_backbone_layers=args.trainable_layers
    )
    
    # Print model info
    model_info = get_model_info(model)
    print("Model Info:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    # Move model to device
    model.to(device)
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Track best validation accuracy
    best_accuracy = 0.0
    
    # Train model
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
    # Save training info
    training_metrics = []
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate model
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            accuracy = evaluate(model, val_dataloader, device)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
                
            # Save metrics for this epoch
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'accuracy': accuracy
            }
            training_metrics.append(epoch_metrics)
        
        # Save checkpoint
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'accuracy': best_accuracy
            }, os.path.join(output_dir, f"checkpoint_{epoch:03d}.pth"))
        
        # Create visualizations
        if epoch % args.vis_freq == 0 or epoch == args.epochs - 1:
            # Get a batch of data
            imgs, targets = next(iter(val_dataloader))
            
            # Save visualizations
            vis_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch:03d}")
            os.makedirs(vis_dir, exist_ok=True)
            
            save_detection_visualization(
                model=model,
                dataset=val_dataloader.dataset,
                images=imgs,
                targets=targets,
                output_dir=vis_dir,
                num_samples=min(len(imgs), 5)
            )
    
    # Calculate training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    # Save training metrics to JSON
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {os.path.join(output_dir, 'best_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detection model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--trainable-layers", type=int, default=3,
                       help="Number of trainable layers in backbone")
    
    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--dataset-type", type=str, default="small",
                      choices=["mini", "small", "full"],
                      help="Type of dataset to use (mini: ~300 images, small: ~5K images, full: ~120K images)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--lr-step-size", type=int, default=3,
                       help="Epochs between learning rate decay")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                       help="Learning rate decay factor")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output parameters
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                       help="Epochs between saving checkpoints")
    parser.add_argument("--eval-freq", type=int, default=1,
                       help="Epochs between evaluations")
    parser.add_argument("--vis-freq", type=int, default=5,
                       help="Epochs between visualizations")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 