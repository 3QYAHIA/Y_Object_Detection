#!/usr/bin/env python3
"""
Test script to verify that the dataset and model are set up correctly
"""

import os
import torch
from tqdm import tqdm
import argparse

# Import project modules
from data.voc_dataset import get_voc_dataloader, download_voc_dataset
from models.detector import get_faster_rcnn_model, get_model_info


def test_dataset(args):
    """Test the dataset loading"""
    print(f"Testing dataset loading...")
    
    # Data directory
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Download dataset if needed
    if args.download:
        download_voc_dataset(data_root, args.voc_year)
    
    # Create dataloader
    dataloader = get_voc_dataloader(
        root_dir=data_root,
        year=args.voc_year,
        image_set='train',
        batch_size=2,
        download=False
    )
    
    # Print dataset info
    num_classes = len(dataloader.dataset.categories) + 1  # +1 for background
    print(f"Dataset loaded successfully:")
    print(f"  Type: Pascal VOC {args.voc_year}")
    print(f"  Size: {len(dataloader.dataset)} images")
    print(f"  Classes: {num_classes-1} (plus background)")
    print(f"  Categories: {dataloader.dataset.categories}")
    
    # Test iterating through dataloader
    print("Testing dataloader iteration...")
    try:
        for i, (images, targets) in enumerate(tqdm(dataloader, total=min(5, len(dataloader)))):
            if i >= 4:  # Only test 5 batches
                break
                
            # Check images and targets
            print(f"\nBatch {i+1}:")
            print(f"  Images: {len(images)}")
            print(f"  Image shape: {images[0].shape}")
            print(f"  Targets: {len(targets)}")
            
            # Print target info for first image
            if targets and len(targets) > 0:
                t = targets[0]
                print(f"  First target:")
                print(f"    Boxes: {t['boxes'].shape}")
                print(f"    Labels: {t['labels'].shape}")
                if len(t['labels']) > 0:
                    print(f"    Class names: {[dataloader.dataset.categories[l.item()] for l in t['labels'][:3]]}")
        
        print("\nDataloader test completed successfully!")
        return True
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        return False


def test_model(args):
    """Test the model creation and forward pass"""
    print(f"\nTesting model creation...")
    
    # Get device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloader to determine number of classes
    data_root = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dataloader = get_voc_dataloader(
        root_dir=data_root,
        year=args.voc_year,
        image_set='train',
        batch_size=2,
        download=False
    )
    num_classes = len(dataloader.dataset.categories) + 1  # +1 for background
    
    # Create model
    model = get_faster_rcnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
        trainable_backbone_layers=3
    )
    
    # Print model info
    model_info = get_model_info(model)
    print("Model created successfully:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"  Number of Classes: {num_classes}")
    
    # Move model to device
    model.to(device)
    
    # Test forward pass
    print("Testing model forward pass...")
    try:
        # Get a batch of data
        images, targets = next(iter(dataloader))
        
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Training mode forward pass
        loss_dict = model(images, targets)
        print("Training mode (with targets):")
        print(f"  Loss components: {list(loss_dict.keys())}")
        print(f"  Total loss: {sum(loss for loss in loss_dict.values()).item():.4f}")
        
        # Inference mode forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        print("Inference mode (without targets):")
        print(f"  Output components: {list(outputs[0].keys())}")
        print(f"  Predictions: {len(outputs[0]['boxes'])}")
        
        print("\nModel test completed successfully!")
        return True
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test object detection setup")
    
    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Data directory")
    parser.add_argument("--download", action="store_true",
                      help="Download dataset if not found")
    parser.add_argument("--voc-year", type=str, default="2012",
                      choices=["2007", "2008", "2009", "2010", "2011", "2012"],
                      help="Pascal VOC dataset year")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50", 
                        choices=["resnet50", "mobilenet_v2"],
                        help="Backbone architecture for the model")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for testing (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Run tests
    dataset_ok = test_dataset(args)
    if dataset_ok:
        model_ok = test_model(args)
    else:
        model_ok = False
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Dataset: {'✓' if dataset_ok else '✗'}")
    print(f"Model: {'✓' if model_ok else '✗'}")
    print(f"Overall: {'PASSED' if dataset_ok and model_ok else 'FAILED'}")


if __name__ == "__main__":
    main() 