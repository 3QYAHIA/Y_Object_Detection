#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import subprocess
import time

# Import project modules
from models.detector import get_faster_rcnn_model, get_model_info
from evaluation.metrics import calculate_iou_matrix

def get_model_stats(backbone):
    """
    Get model statistics (parameters, memory usage, etc.)
    
    Args:
        backbone: Name of backbone (resnet50 or mobilenet_v2)
        
    Returns:
        stats: Dictionary of model statistics
    """
    # Create model with smaller number of classes for tiny subset
    model = get_faster_rcnn_model(
        num_classes=6,  # 5 classes + background for tiny subset
        backbone=backbone,
        pretrained=True
    )
    
    # Get model info
    model_info = get_model_info(model)
    
    # Get additional metrics
    stats = {
        'backbone': backbone,
        'total_parameters': model_info['total_parameters'],
        'trainable_parameters': model_info['trainable_parameters'],
    }
    
    return stats

def load_metrics(backbone, metric_type):
    """
    Load metrics from output directory
    
    Args:
        backbone: Name of backbone (resnet50 or mobilenet_v2)
        metric_type: Type of metric to load (mAP, precision_recall_f1, speed)
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Define file paths
    if metric_type == 'mAP':
        file_path = os.path.join("outputs", backbone, "evaluation", "mAP_results.json")
    elif metric_type == 'precision_recall_f1':
        file_path = os.path.join("outputs", backbone, "evaluation", "precision_recall_f1.csv")
    elif metric_type == 'speed':
        file_path = os.path.join("outputs", backbone, "evaluation", "speed_metrics.json")
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    # Load metrics
    if metric_type == 'precision_recall_f1':
        # Load CSV file
        df = pd.read_csv(file_path)
        return df
    else:
        # Load JSON file
        with open(file_path, 'r') as f:
            metrics = json.load(f)
        return metrics

def compare_model_stats():
    """
    Compare model statistics (parameters, memory usage, etc.)
    
    Returns:
        df: DataFrame with model statistics
    """
    # Get model stats
    resnet_stats = get_model_stats('resnet50')
    mobilenet_stats = get_model_stats('mobilenet_v2')
    
    # Create DataFrame
    stats = [resnet_stats, mobilenet_stats]
    df = pd.DataFrame(stats)
    
    return df

def plot_model_stats(df, output_dir):
    """
    Plot model statistics
    
    Args:
        df: DataFrame with model statistics
        output_dir: Directory to save plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='total_parameters', data=df)
    plt.title('Total Parameters by Backbone')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')
    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.total_parameters, f"{row.total_parameters:,}", 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_parameters.png'))
    plt.close()
    
    # Trainable parameters plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='trainable_parameters', data=df)
    plt.title('Trainable Parameters by Backbone')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')
    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.trainable_parameters, f"{row.trainable_parameters:,}", 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trainable_parameters.png'))
    plt.close()

def compare_speed_metrics():
    """
    Compare speed metrics (FPS, latency)
    
    Returns:
        df: DataFrame with speed metrics
    """
    # Load speed metrics
    resnet_speed = load_metrics('resnet50', 'speed')
    mobilenet_speed = load_metrics('mobilenet_v2', 'speed')
    
    # Create DataFrame
    speeds = [
        {'backbone': 'resnet50', 'fps': resnet_speed['fps'], 'latency': resnet_speed['latency']},
        {'backbone': 'mobilenet_v2', 'fps': mobilenet_speed['fps'], 'latency': mobilenet_speed['latency']}
    ]
    df = pd.DataFrame(speeds)
    
    return df

def plot_speed_metrics(df, output_dir):
    """
    Plot speed metrics
    
    Args:
        df: DataFrame with speed metrics
        output_dir: Directory to save plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # FPS plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='fps', data=df)
    plt.title('Frames Per Second by Backbone')
    plt.ylabel('FPS')
    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.fps, f"{row.fps:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fps.png'))
    plt.close()
    
    # Latency plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='latency', data=df)
    plt.title('Inference Latency by Backbone')
    plt.ylabel('Latency (ms)')
    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.latency, f"{row.latency:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency.png'))
    plt.close()

def compare_map_metrics():
    """
    Compare mAP metrics
    
    Returns:
        df: DataFrame with mAP metrics
    """
    # Load mAP metrics
    resnet_map = load_metrics('resnet50', 'mAP')
    mobilenet_map = load_metrics('mobilenet_v2', 'mAP')
    
    # Create DataFrames
    resnet_df = pd.DataFrame({'backbone': 'resnet50', 'metric': list(resnet_map.keys()), 'value': list(resnet_map.values())})
    mobilenet_df = pd.DataFrame({'backbone': 'mobilenet_v2', 'metric': list(mobilenet_map.keys()), 'value': list(mobilenet_map.values())})
    
    # Combine DataFrames
    df = pd.concat([resnet_df, mobilenet_df])
    
    # Filter metrics
    metrics_to_plot = ['mAP', 'mAP_0.5', 'mAP_0.75']
    df = df[df['metric'].isin(metrics_to_plot)]
    
    return df

def plot_map_metrics(df, output_dir):
    """
    Plot mAP metrics
    
    Args:
        df: DataFrame with mAP metrics
        output_dir: Directory to save plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # mAP plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='metric', y='value', hue='backbone', data=df)
    plt.title('Mean Average Precision by Backbone')
    plt.ylabel('mAP')
    plt.xlabel('Metric')
    
    # Add text labels
    for i, row in enumerate(df.itertuples()):
        plt.text(i % 3 - 0.2 if row.backbone == 'resnet50' else i % 3 + 0.2, 
                 row.value, f"{row.value:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_metrics.png'))
    plt.close()

def compare_precision_recall_f1():
    """
    Compare precision, recall, F1 metrics
    
    Returns:
        resnet_df: DataFrame with ResNet metrics
        mobilenet_df: DataFrame with MobileNet metrics
    """
    # Load metrics
    resnet_df = load_metrics('resnet50', 'precision_recall_f1')
    mobilenet_df = load_metrics('mobilenet_v2', 'precision_recall_f1')
    
    # Add backbone column
    resnet_df['backbone'] = 'resnet50'
    mobilenet_df['backbone'] = 'mobilenet_v2'
    
    return resnet_df, mobilenet_df

def plot_precision_recall_f1(resnet_df, mobilenet_df, output_dir):
    """
    Plot precision, recall, F1 metrics
    
    Args:
        resnet_df: DataFrame with ResNet metrics
        mobilenet_df: DataFrame with MobileNet metrics
        output_dir: Directory to save plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape DataFrames for plotting
    metrics = ['precision', 'recall', 'f1_score']
    
    # ResNet reshape
    resnet_data = []
    for i, row in resnet_df.iterrows():
        for metric in metrics:
            resnet_data.append({
                'backbone': 'resnet50',
                'category': row['category'],
                'metric': metric,
                'value': row[metric]
            })
    resnet_plot_df = pd.DataFrame(resnet_data)
    
    # MobileNet reshape
    mobilenet_data = []
    for i, row in mobilenet_df.iterrows():
        for metric in metrics:
            mobilenet_data.append({
                'backbone': 'mobilenet_v2',
                'category': row['category'],
                'metric': metric,
                'value': row[metric]
            })
    mobilenet_plot_df = pd.DataFrame(mobilenet_data)
    
    # Combine DataFrames
    plot_df = pd.concat([resnet_plot_df, mobilenet_plot_df])
    
    # Get unique categories
    categories = plot_df['category'].unique()
    
    # Plot per category
    for category in categories:
        cat_df = plot_df[plot_df['category'] == category]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='metric', y='value', hue='backbone', data=cat_df)
        plt.title(f'Metrics for {category}')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        
        # Add text labels
        for i, row in enumerate(cat_df.itertuples()):
            plt.text(i % 3 - 0.2 if row.backbone == 'resnet50' else i % 3 + 0.2, 
                     row.value, f"{row.value:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_{category}.png'))
        plt.close()
    
    # Plot average metrics
    avg_df = plot_df.groupby(['backbone', 'metric'])['value'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='metric', y='value', hue='backbone', data=avg_df)
    plt.title('Average Metrics by Backbone')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    
    # Add text labels
    for i, row in enumerate(avg_df.itertuples()):
        plt.text(i % 3 - 0.2 if row.backbone == 'resnet50' else i % 3 + 0.2, 
                 row.value, f"{row.value:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_metrics.png'))
    plt.close()

def create_comparison_table():
    """
    Create comparison table
    
    Returns:
        df: DataFrame with comparison metrics
    """
    # Get model stats
    model_stats = compare_model_stats()
    
    # Get speed metrics
    speed_metrics = compare_speed_metrics()
    
    # Get mAP metrics
    map_metrics = compare_map_metrics()
    
    # Create results dictionary
    results = {}
    
    for backbone in ['resnet50', 'mobilenet_v2']:
        backbone_results = {
            'backbone': backbone,
            'total_parameters': model_stats[model_stats['backbone'] == backbone]['total_parameters'].values[0],
            'trainable_parameters': model_stats[model_stats['backbone'] == backbone]['trainable_parameters'].values[0],
            'fps': speed_metrics[speed_metrics['backbone'] == backbone]['fps'].values[0],
            'latency': speed_metrics[speed_metrics['backbone'] == backbone]['latency'].values[0],
        }
        
        # Add mAP metrics
        for metric in ['mAP', 'mAP_0.5', 'mAP_0.75']:
            value = map_metrics[(map_metrics['backbone'] == backbone) & (map_metrics['metric'] == metric)]['value'].values[0]
            backbone_results[metric] = value
        
        results[backbone] = backbone_results
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.reset_index(drop=True, inplace=True)
    
    return df

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up output directory
    output_dir = os.path.join("outputs", "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if both backbones have been evaluated
    for backbone in ['resnet50', 'mobilenet_v2']:
        eval_dir = os.path.join("outputs", backbone, "evaluation")
        if not os.path.exists(eval_dir):
            # If evaluation directory doesn't exist, run evaluation
            print(f"Evaluation for {backbone} not found. Running evaluation...")
            cmd = [
                "python", "evaluate.py",
                "--backbone", backbone,
                "--batch-size", str(args.batch_size),
                "--num-test-images", str(args.num_test_images),
                "--conf-threshold", str(args.conf_threshold),
                "--device", args.device
            ]
            if args.tiny:
                cmd.append("--tiny")
            
            subprocess.run(cmd)
    
    print("Comparing model statistics...")
    model_stats = compare_model_stats()
    plot_model_stats(model_stats, output_dir)
    
    print("Comparing speed metrics...")
    speed_metrics = compare_speed_metrics()
    plot_speed_metrics(speed_metrics, output_dir)
    
    print("Comparing mAP metrics...")
    map_metrics = compare_map_metrics()
    plot_map_metrics(map_metrics, output_dir)
    
    print("Comparing precision, recall, F1 metrics...")
    resnet_pr_f1, mobilenet_pr_f1 = compare_precision_recall_f1()
    plot_precision_recall_f1(resnet_pr_f1, mobilenet_pr_f1, output_dir)
    
    print("Creating comparison table...")
    comparison_table = create_comparison_table()
    comparison_table.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
    
    # Print summary
    print("\nComparison Summary:")
    print(comparison_table.to_string(index=False))
    
    print("\nComparison complete. Results saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare object detection backbones")
    
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
    parser.add_argument("--subset", action="store_true",
                       help="Use subset of COCO dataset (10 classes)")
    parser.add_argument("--tiny", action="store_true", default=True,
                       help="Use tiny subset of COCO dataset (5 classes, <300MB)")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args) 