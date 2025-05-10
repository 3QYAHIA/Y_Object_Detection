#!/usr/bin/env python3
import os
import argparse
import subprocess
import torch

def check_gpu():
    """
    Check if GPU is available
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU available. Using CPU.")
        return False

def main(args):
    """
    Main entry point for the project
    
    Args:
        args: Command line arguments
    """
    # Check GPU availability
    has_gpu = check_gpu()
    device = "cuda" if has_gpu else "cpu"
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Determine dataset related arguments
    dataset_args = []
    if args.dataset == "voc":
        print(f"\nUsing Pascal VOC {args.voc_year} dataset")
        dataset_args = [
            "--dataset", "voc",
            "--voc-year", args.voc_year,
            "--voc-train-set", args.voc_train_set,
            "--voc-val-set", args.voc_val_set
        ]
        if args.download:
            dataset_args.append("--download")
    else:  # COCO dataset
        dataset_type = "mini" if args.tiny else args.dataset_type
        print(f"\nUsing COCO dataset (type: {dataset_type})")
        dataset_args = [
            "--dataset", "coco",
            "--dataset-type", dataset_type
        ]
        if args.download:
            dataset_args.append("--download")
            
    # Step 1: Download dataset if requested
    if args.download:
        if args.dataset == "voc":
            print(f"\n=== Step 1: Setting up Pascal VOC {args.voc_year} dataset ===")
        else:
            print(f"\n=== Step 1: Setting up COCO dataset ({dataset_type}) ===")
    
    # Step 2: Train models
    if args.train:
        print("\n=== Step 2: Training models ===")
        
        # Train ResNet-50 model
        print("\nTraining ResNet-50 model...")
        subprocess.run([
            "python", "train.py",
            "--backbone", "resnet50",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", device,
        ] + dataset_args)
        
        # Train MobileNetV2 model
        print("\nTraining MobileNetV2 model...")
        subprocess.run([
            "python", "train.py",
            "--backbone", "mobilenet_v2",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", device,
        ] + dataset_args)
    
    # Step 3: Evaluate models
    if args.evaluate:
        print("\n=== Step 3: Evaluating models ===")
        
        # Evaluate ResNet-50 model
        print("\nEvaluating ResNet-50 model...")
        subprocess.run([
            "python", "evaluate.py",
            "--backbone", "resnet50",
            "--batch-size", str(args.batch_size),
            "--device", device,
        ] + dataset_args)
        
        # Evaluate MobileNetV2 model
        print("\nEvaluating MobileNetV2 model...")
        subprocess.run([
            "python", "evaluate.py",
            "--backbone", "mobilenet_v2",
            "--batch-size", str(args.batch_size),
            "--device", device,
        ] + dataset_args)
    
    # Step 4: Compare models
    if args.compare:
        print("\n=== Step 4: Comparing models ===")
        subprocess.run([
            "python", "compare_backbones.py",
            "--batch-size", str(args.batch_size),
            "--device", device,
        ] + dataset_args)
    
    # Step 5: Run inference
    if args.inference and args.image:
        print("\n=== Step 5: Running inference ===")
        
        # Choose backbone
        backbone = args.backbone if args.backbone else "resnet50"
        
        subprocess.run([
            "python", "detect.py",
            "--backbone", backbone,
            "--input", args.image,
            "--device", device,
        ] + dataset_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with CNN Architectures")
    
    # Main flags
    parser.add_argument("--download", action="store_true", help="Download dataset if not found")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "voc"],
                       help="Dataset to use (coco or pascal voc)")
    
    # COCO dataset parameters
    parser.add_argument("--dataset-type", type=str, default="small", 
                      choices=["mini", "small", "full"],
                      help="Type of COCO dataset (mini: ~300 images, small: ~5K images, full: ~120K images)")
    parser.add_argument("--tiny", action="store_true", default=False, 
                       help="Use tiny subset of COCO dataset (equivalent to --dataset-type mini)")
    
    # Pascal VOC parameters
    parser.add_argument("--voc-year", type=str, default="2012",
                      choices=["2007", "2008", "2009", "2010", "2011", "2012"],
                      help="Pascal VOC dataset year")
    parser.add_argument("--voc-train-set", type=str, default="train",
                      choices=["train", "trainval"],
                      help="Pascal VOC train image set")
    parser.add_argument("--voc-val-set", type=str, default="val",
                      choices=["val", "test"],
                      help="Pascal VOC validation image set")
    
    # Training/evaluation parameters
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet_v2"], 
                      help="Backbone architecture for the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and evaluation")
    
    # Inference parameters
    parser.add_argument("--image", type=str, help="Path to input image or directory for inference")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any([args.download, args.train, args.evaluate, args.compare, args.inference]):
        parser.print_help()
    else:
        main(args) 