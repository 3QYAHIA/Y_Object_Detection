#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# Import project modules
from data.coco_dataset import get_coco_dataloader
from models.detector import get_faster_rcnn_model, get_model_info
from utils.visualization import save_detection_visualization

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train model for one epoch
    
    Args:
        model: Detection model
        optimizer: Optimizer
        data_loader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch
        print_freq: Frequency to print loss
        
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
        if i % print_freq == 0:
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

@torch.inference_mode()
def evaluate(model, data_loader, device):
    """
    Evaluate model on validation data
    
    Args:
        model: Detection model
        data_loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        predictions: List of model predictions
    """
    # Set model to evaluation mode
    model.eval()
    
    # Collect predictions
    predictions = []
    
    # Iterate over batches
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        # Move to device
        images = list(image.to(device) for image in images)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        
        # Collect predictions
        for i, output in enumerate(outputs):
            # Get image id
            image_id = targets[i]['image_id']
            
            # Add to predictions
            predictions.append((
                output['boxes'],
                output['labels'],
                output['scores'],
                image_id
            ))
    
    return predictions

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
    
    # Set up data paths - always use tiny_subset if it exists
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
    
    # Create dataloaders
    train_dataloader = get_coco_dataloader(
        root_dir=root_dir,
        ann_file=ann_file,
        batch_size=args.batch_size,
        train=True,
        subset=args.subset or args.tiny
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
    print(f"  Backbone: {model_info['backbone']}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    
    # Save model info
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=4)
    
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
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Track best validation loss
    best_loss = float('inf')
    
    # Train model
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
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
        
        # Save model at checkpoint frequency
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            # Save checkpoint
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch:03d}.pth"))
        
        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        
        # Log metrics to TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        
        # Log learning rate
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Run inference on a few training samples
        if epoch % args.visualization_freq == 0 or epoch == args.epochs - 1:
            # Get a batch of data
            batch_imgs, batch_targets = next(iter(train_dataloader))
            
            # Save visualizations
            vis_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch:03d}")
            save_detection_visualization(
                model=model,
                dataset=train_dataloader.dataset,
                images=batch_imgs,
                targets=batch_targets,
                output_dir=vis_dir,
                num_samples=args.num_vis_samples
            )
    
    # Calculate training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object detection model")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    parser.add_argument("--trainable-layers", type=int, default=3,
                       help="Number of trainable layers in backbone")
    
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
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--subset", action="store_true",
                       help="Use subset of COCO dataset (10 classes)")
    parser.add_argument("--tiny", action="store_true", default=True,
                       help="Use tiny subset of COCO dataset (5 classes, <300MB)")
    
    # Output parameters
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                       help="Epochs between saving checkpoints")
    parser.add_argument("--visualization-freq", type=int, default=5,
                       help="Epochs between creating visualizations")
    parser.add_argument("--num-vis-samples", type=int, default=5,
                       help="Number of samples to visualize")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 