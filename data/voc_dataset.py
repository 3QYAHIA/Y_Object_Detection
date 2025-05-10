import os
import torch
import torchvision
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# VOC class names (20 classes)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(VOCDetection):
    """
    Custom Pascal VOC dataset with additional functionality for object detection
    
    This dataset adapts the torchvision VOCDetection dataset to provide
    a consistent interface with our detection model requirements
    """
    def __init__(self, root, year="2012", image_set="train", transform=None, download=False):
        """
        Initialize VOC dataset
        
        Args:
            root: Root directory of the VOC dataset
            year: Dataset year ("2007" to "2012")
            image_set: Image set to use ("train", "trainval", "val" or "test" for 2007)
            transform: Transform to apply to images
            download: Whether to download the dataset if not found
        """
        super(VOCDataset, self).__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download
        )
        self.transform = transform
        
        # Create a mapping from class names to indices (starting from 1, as 0 is background)
        self.class_to_idx = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}
        
    def __getitem__(self, index):
        """
        Get image and annotations
        
        Args:
            index: Index of image
            
        Returns:
            img: Image tensor
            target: Target dictionary with boxes, labels, etc.
        """
        # Get image and annotations
        img, annotation = super(VOCDataset, self).__getitem__(index)
        
        # Initialize target dictionary
        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([index])
        }
        
        # Get image id (filename without extension)
        img_id = Path(self.images[index]).stem
        target['image_id'] = torch.tensor([int(img_id) if img_id.isdigit() else index])
        
        # Process objects
        if 'annotation' in annotation and 'object' in annotation['annotation']:
            for obj in annotation['annotation']['object']:
                # Extract class name
                class_name = obj['name']
                
                # Skip if class not in our classes
                if class_name not in self.class_to_idx:
                    continue
                
                # Extract bounding box
                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
                
                # Skip invalid boxes
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # Add to target
                target['boxes'].append([xmin, ymin, xmax, ymax])
                target['labels'].append(self.class_to_idx[class_name])
        
        # Skip images without annotations
        if len(target['boxes']) == 0:
            # Return a dummy target for empty annotations
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0), dtype=torch.int64)
            return img, target
        
        # Convert lists to tensors
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        
        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    @property
    def categories(self):
        """Return a dictionary mapping from category id to category name"""
        return {i+1: name for i, name in enumerate(VOC_CLASSES)}

def collate_fn(batch):
    """
    Collate function for batch creation
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        images: List of image tensors
        targets: List of target dictionaries
    """
    # Separate images and targets
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    return images, targets

def download_voc_dataset(data_dir, year="2012"):
    """
    Download Pascal VOC dataset
    
    Args:
        data_dir: Directory to save dataset
        year: Dataset year ("2007" to "2012")
    """
    os.makedirs(data_dir, exist_ok=True)
    voc_dir = os.path.join(data_dir, "voc")
    os.makedirs(voc_dir, exist_ok=True)
    
    print(f"Setting up Pascal VOC {year} dataset...")
    # Use VOCDetection download functionality
    VOCDetection(root=voc_dir, year=year, image_set="trainval", download=True)
    print("Dataset setup complete.")

def get_voc_dataloader(root_dir, year="2012", image_set="train", batch_size=4, num_workers=4, download=False):
    """
    Create Pascal VOC dataloader
    
    Args:
        root_dir: Directory with VOC data
        year: Dataset year ("2007" to "2012")
        image_set: Image set to use ("train", "trainval", "val" or "test" for 2007)
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        download: Whether to download the dataset if not found
        
    Returns:
        dataloader: VOC dataloader
    """
    # Define transforms
    if image_set in ["train", "trainval"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = VOCDataset(
        root=root_dir,
        year=year,
        image_set=image_set,
        transform=transform,
        download=download
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set in ["train", "trainval"]),
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader 