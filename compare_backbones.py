#!/usr/bin/env python3
# Set matplotlib backend to non-interactive to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import torch
import numpy as np
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import subprocess
from pathlib import Path

# Import project modules
from data.coco_dataset import get_coco_dataloader
from data.voc_dataset import get_voc_dataloader, VOC_CLASSES
from models.detector import get_faster_rcnn_model, get_model_info
from evaluate import calculate_metrics

def evaluate_speed(model, device, input_size=(800, 800), num_iterations=50):
    """
    Evaluate model inference speed
    
    Args:
        model: Model to evaluate
        device: Device to run on
        input_size: Input image size (height, width)
        num_iterations: Number of iterations to run
        
    Returns:
        fps: Frames per second
        latency: Latency in milliseconds
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor (Faster R-CNN expects a list of tensors)
    dummy_input = [torch.randn(3, input_size[0], input_size[1]).to(device)]
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    # Synchronize after timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time
    latency = (elapsed_time / num_iterations) * 1000  # convert to ms
    
    return fps, latency

def load_model_metrics(backbone):
    """
    Load evaluation metrics for a model
    
    Args:
        backbone: Backbone name
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Build path to metrics files
    output_dir = os.path.join("outputs", backbone, "evaluation")
    
    # Initialize metrics
    metrics = {
        "backbone": backbone,
        "mAP": 0,
        "mAP_0.5": 0,
        "mAP_0.75": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0
    }
    
    # Load mAP results
    map_file = os.path.join(output_dir, "map_results.json")
    if os.path.exists(map_file):
        try:
            with open(map_file, "r") as f:
                map_results = json.load(f)
                metrics["mAP"] = map_results.get("mAP", 0)
                metrics["mAP_0.5"] = map_results.get("mAP_0.5", 0)
                metrics["mAP_0.75"] = map_results.get("mAP_0.75", 0)
        except Exception as e:
            print(f"Error loading mAP results for {backbone}: {e}")
    
    # Load precision, recall, F1 results
    pr_file = os.path.join(output_dir, "precision_recall_f1.csv")
    if os.path.exists(pr_file):
        try:
            df = pd.read_csv(pr_file)
            if not df.empty:
                # Calculate weighted average by support
                weighted_precision = (df["precision"] * df["support"]).sum() / df["support"].sum()
                weighted_recall = (df["recall"] * df["support"]).sum() / df["support"].sum()
                weighted_f1 = (df["f1_score"] * df["support"]).sum() / df["support"].sum()
                
                metrics["precision"] = weighted_precision
                metrics["recall"] = weighted_recall
                metrics["f1_score"] = weighted_f1
        except Exception as e:
            print(f"Error loading precision-recall results for {backbone}: {e}")
    
    return metrics

def benchmark_inference(model, data_loader, device, num_runs=50):
    """
    Benchmark inference time
    
    Args:
        model: Detection model
        data_loader: DataLoader for test data
        device: Device to run on
        num_runs: Number of runs to average
        
    Returns:
        avg_time: Average inference time per image
    """
    model.eval()
    
    # Get first batch
    try:
        images, _ = next(iter(data_loader))
    except StopIteration:
        # If dataloader is empty, return 0
        return 0
    
    # Make sure we're using a single image for benchmarking
    if len(images) > 1:
        images = [images[0]]
    
    # Convert to device
    images = [image.to(device) for image in images]
    
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            _ = model(images)
    
    # Run benchmark
    times = []
    
    for _ in range(num_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(images)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    return avg_time

def compare_backbones(backbones, args):
    """
    Compare different backbones
    
    Args:
        backbones: List of backbones to compare
        args: Command line arguments
        
    Returns:
        results: Dictionary of results
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join("evaluation", "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data
    print("Setting up dataset...")
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Set up dataloaders based on dataset type
    if args.dataset == "voc":
        # Create dataloader for Pascal VOC
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
    
    # Initialize results
    results = {
        'backbone': [],
        'mAP_0.5': [],
        'mAP_0.75': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'inference_time': [],
        'model_size': [],
        'params_total': [],
        'params_trainable': []
    }
    
    # Compare each backbone
    for backbone in backbones:
        print(f"\nEvaluating {backbone} backbone...")
        
        # Create model
        model = get_faster_rcnn_model(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )
        
        # Get model info
        model_info = get_model_info(model)
        print(f"Model info:")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Load model weights
        model_path = os.path.join("outputs", backbone, "best_model.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No saved model found at {model_path}, using untrained model")
        
        # Move model to device
        model.to(device)
        
        # Benchmark inference time
        print("Benchmarking inference time...")
        inference_time = benchmark_inference(model, dataloader, device, num_runs=args.num_runs)
        print(f"Average inference time: {inference_time * 1000:.2f} ms per image")
        
        # Calculate model size
        model_size = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        print(f"Model size: {model_size:.2f} MB")
        
        # Calculate metrics
        print("Calculating metrics...")
        backbone_output_dir = os.path.join(output_dir, backbone)
        os.makedirs(backbone_output_dir, exist_ok=True)
        
        metrics = calculate_metrics(model, dataloader, device, num_classes, dataset_type, backbone_output_dir)
        
        # Add to results
        results['backbone'].append(backbone)
        results['mAP_0.5'].append(metrics['mAP'].get('mAP@0.5', 0))
        results['mAP_0.75'].append(metrics['mAP'].get('mAP@0.75', 0))
        results['precision'].append(metrics['overall']['precision'])
        results['recall'].append(metrics['overall']['recall'])
        results['f1'].append(metrics['overall']['f1'])
        results['inference_time'].append(inference_time * 1000)  # Convert to ms
        results['model_size'].append(model_size)
        results['params_total'].append(model_info['total_parameters'])
        results['params_trainable'].append(model_info['trainable_parameters'])
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'backbone_comparison.csv'), index=False)
    
    # Create visualization
    create_comparison_charts(results_df, output_dir)
    
    return results_df

def create_comparison_charts(results_df, output_dir):
    """
    Create charts to compare backbones
    
    Args:
        results_df: DataFrame with results
        output_dir: Directory to save charts
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy metrics
    axs[0, 0].bar(results_df['backbone'], results_df['mAP_0.5'], color='steelblue')
    axs[0, 0].set_title('mAP@0.5')
    axs[0, 0].set_ylim([0, 1])
    
    # Precision, recall, F1
    width = 0.25
    x = np.arange(len(results_df['backbone']))
    
    axs[0, 1].bar(x - width, results_df['precision'], width, label='Precision', color='lightcoral')
    axs[0, 1].bar(x, results_df['recall'], width, label='Recall', color='lightgreen')
    axs[0, 1].bar(x + width, results_df['f1'], width, label='F1', color='skyblue')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(results_df['backbone'])
    axs[0, 1].set_title('Precision, Recall, F1')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()
    
    # Inference time
    axs[1, 0].bar(results_df['backbone'], results_df['inference_time'], color='orange')
    axs[1, 0].set_title('Inference Time (ms)')
    
    # Model size
    axs[1, 1].bar(results_df['backbone'], results_df['params_total'] / 1e6, color='mediumseagreen')
    axs[1, 1].set_title('Model Parameters (millions)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'backbone_comparison.png'))
    plt.close()

def main(args):
    """
    Main function for comparing backbones
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Backbones to compare
    backbones = ["resnet50", "mobilenet_v2"]
    
    # Compare backbones
    results_df = compare_backbones(backbones, args)
    
    # Print summary
    print("\nBackbone Comparison Summary:")
    print(results_df.to_string(index=False))
    
    # Calculate relative performance
    if len(results_df) > 1:
        base_backbone = "resnet50"
        compare_backbone = "mobilenet_v2"
        
        base_idx = results_df[results_df['backbone'] == base_backbone].index[0]
        compare_idx = results_df[results_df['backbone'] == compare_backbone].index[0]
        
        speed_diff = results_df.loc[base_idx, 'inference_time'] / results_df.loc[compare_idx, 'inference_time']
        size_diff = results_df.loc[base_idx, 'params_total'] / results_df.loc[compare_idx, 'params_total']
        acc_diff = results_df.loc[compare_idx, 'mAP_0.5'] / results_df.loc[base_idx, 'mAP_0.5']
        
        print(f"\nRelative Performance ({compare_backbone} vs {base_backbone}):")
        print(f"  Speed: {speed_diff:.2f}x faster")
        print(f"  Size: {size_diff:.2f}x smaller")
        print(f"  Accuracy: {acc_diff:.2f}x relative accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare object detection backbones")
    
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
    
    # Benchmark parameters
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--num-runs", type=int, default=50,
                       help="Number of runs for benchmarking")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 