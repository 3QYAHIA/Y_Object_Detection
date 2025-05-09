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

# Import project modules
from data.coco_dataset import get_coco_dataloader
from models.detector import get_faster_rcnn_model, get_model_info

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

def compare_models(args):
    """
    Compare models with different backbones
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join("outputs", "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get models information
    backbones = ['resnet50', 'mobilenet_v2']
    backbone_info = {}
    model_metrics = []
    
    for backbone in backbones:
        print(f"\nComparing backbone: {backbone}")
        
        # Create model
        model = get_faster_rcnn_model(
            num_classes=91,  # COCO has 80 classes + background
            backbone=backbone,
            pretrained=False,
            trainable_backbone_layers=0
        )
        
        # Get model info
        model_info = get_model_info(model)
        backbone_info[backbone] = model_info
        
        print(f"Model info:")
        print(f"  Total parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Move model to device
        model.to(device)
        
        # Measure inference speed
        print("Measuring inference speed...")
        fps, latency = evaluate_speed(model, device)
        print(f"  FPS: {fps:.2f}")
        print(f"  Latency: {latency:.2f} ms")
        
        # Check if trained model exists, otherwise we'll just compare architectures
        model_path = os.path.join("outputs", backbone, "best_model.pth")
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No trained model found for {backbone}. Skipping accuracy comparison.")
        
        # Run evaluation if model exists and evaluation metrics file doesn't exist
        eval_dir = os.path.join("outputs", backbone, "evaluation")
        map_file = os.path.join(eval_dir, "map_results.json")
        
        if os.path.exists(model_path) and not os.path.exists(map_file):
            print(f"Running evaluation for {backbone}...")
            cmd = f"python evaluate.py --backbone {backbone} --device {args.device}"
            subprocess.run(cmd, shell=True)
        
        # Load evaluation metrics
        metrics = load_model_metrics(backbone)
        
        # Add speed metrics
        metrics["fps"] = fps
        metrics["latency"] = latency
        metrics["parameters"] = model_info["total_parameters"]
        
        # Add to metrics list
        model_metrics.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(model_metrics)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, "backbone_comparison.csv"), index=False)
    
    # Generate comparison plots
    generate_comparison_plots(comparison_df, output_dir)
    
    # Print comparison table
    print("\nBackbone Comparison:")
    print(comparison_df.to_string(index=False))

def generate_comparison_plots(df, output_dir):
    """
    Generate comparison plots
    
    Args:
        df: DataFrame with comparison metrics
        output_dir: Output directory
    """
    # Set style
    sns.set(style="whitegrid")
    
    # Plot speed comparison
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)
    sns.barplot(x="backbone", y="fps", data=df, ax=ax)
    plt.title("Inference Speed (FPS)")
    plt.ylabel("Frames Per Second")
    
    ax = plt.subplot(122)
    sns.barplot(x="backbone", y="latency", data=df, ax=ax)
    plt.title("Inference Latency")
    plt.ylabel("Latency (ms)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_comparison.png"))
    plt.close()
    
    # Plot accuracy metrics
    plt.figure(figsize=(12, 6))
    
    # mAP comparison
    ax = plt.subplot(121)
    metrics = ["mAP_0.5", "mAP_0.75", "mAP"]
    for i, backbone in enumerate(df["backbone"]):
        values = [df[df["backbone"] == backbone][metric].values[0] for metric in metrics]
        ax.bar(np.arange(len(metrics)) + 0.2*i, values, width=0.2, label=backbone)
    
    ax.set_xticks(np.arange(len(metrics)) + 0.1)
    ax.set_xticklabels(["mAP@0.5", "mAP@0.75", "mAP"])
    plt.title("mAP Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    
    # Precision, recall, F1 comparison
    ax = plt.subplot(122)
    metrics = ["precision", "recall", "f1_score"]
    for i, backbone in enumerate(df["backbone"]):
        values = [df[df["backbone"] == backbone][metric].values[0] for metric in metrics]
        ax.bar(np.arange(len(metrics)) + 0.2*i, values, width=0.2, label=backbone)
    
    ax.set_xticks(np.arange(len(metrics)) + 0.1)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    plt.title("Classification Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()
    
    # Size vs Performance plot
    plt.figure(figsize=(10, 6))
    
    # Create bubble chart of parameters vs mAP with FPS as size
    sizes = df["fps"] * 20  # Scale for visualization
    
    for i, row in df.iterrows():
        plt.scatter(
            row["parameters"] / 1e6,  # Parameters in millions
            row["mAP"],
            s=sizes[i],
            alpha=0.7,
            label=row["backbone"]
        )
        plt.text(
            row["parameters"] / 1e6,
            row["mAP"],
            f"{row['backbone']}\n{row['fps']:.1f} FPS",
            ha='center',
            va='center'
        )
    
    plt.title("Model Size vs. Accuracy vs. Speed")
    plt.xlabel("Parameters (millions)")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_vs_performance.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare object detection models with different backbones")
    
    # Comparison parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Run comparison
    compare_models(args)

if __name__ == "__main__":
    main() 