#!/usr/bin/env python3
import os
import argparse
import subprocess
import torch

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU available. Using CPU.")
        return False

def main(args):
    """Main entry point for the project"""
    # Check GPU availability
    has_gpu = check_gpu()
    device = "cuda" if has_gpu else "cpu"
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Step 1: Download COCO dataset
    if args.download:
        print("\n=== Step 1: Downloading COCO dataset (tiny subset) ===")
        subprocess.run(["python", "data/download_coco.py"])
    
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
            "--tiny"
        ])
        
        # Train MobileNetV2 model
        print("\nTraining MobileNetV2 model...")
        subprocess.run([
            "python", "train.py",
            "--backbone", "mobilenet_v2",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", device,
            "--tiny"
        ])
    
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
            "--tiny"
        ])
        
        # Evaluate MobileNetV2 model
        print("\nEvaluating MobileNetV2 model...")
        subprocess.run([
            "python", "evaluate.py",
            "--backbone", "mobilenet_v2",
            "--batch-size", str(args.batch_size),
            "--device", device,
            "--tiny"
        ])
    
    # Step 4: Compare models
    if args.compare:
        print("\n=== Step 4: Comparing models ===")
        subprocess.run([
            "python", "compare_backbones.py",
            "--batch-size", str(args.batch_size),
            "--device", device,
            "--tiny"
        ])
    
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
            "--tiny"
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with CNN Architectures")
    
    # Main flags
    parser.add_argument("--download", action="store_true", help="Download COCO dataset (tiny subset)")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--compare", action="store_true", help="Compare models")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    
    # Training/evaluation parameters
    parser.add_argument("--backbone", type=str, choices=["resnet50", "mobilenet_v2"], 
                      help="Backbone architecture for the model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--tiny", action="store_true", default=True, help="Use tiny subset of COCO dataset (<300MB)")
    
    # Inference parameters
    parser.add_argument("--image", type=str, help="Path to input image or directory for inference")
    
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any([args.download, args.train, args.evaluate, args.compare, args.inference]):
        parser.print_help()
    else:
        main(args) 