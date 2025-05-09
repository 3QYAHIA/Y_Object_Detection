import os
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class CocoDataset(CocoDetection):
    """
    Custom COCO dataset with additional functionality
    """
    def __init__(self, root, ann_file, transform=None, subset=False):
        super(CocoDataset, self).__init__(root, ann_file)
        self.transform = transform
        
        # Create dictionary of category names
        self.categories = {}
        for cat in self.coco.cats.values():
            self.categories[cat['id']] = cat['name']
        
        # Filter categories if subset is True
        if subset:
            # If subset, keep only specific categories
            keep_cats = set()
            # Keep first 10 categories for subset
            for i, cat_id in enumerate(self.coco.cats.keys()):
                if i < 10:  # Keep 10 categories
                    keep_cats.add(cat_id)
        
            # Filter annotations to only include the subset categories
            new_ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                # Keep image if it has annotations from the subset categories
                if any(ann['category_id'] in keep_cats for ann in anns):
                    new_ids.append(img_id)
            
            # Update image ids
            self.ids = new_ids
            
            # Update categories
            self.categories = {k: v for k, v in self.categories.items() if k in keep_cats}
        
        # Create a mapping from category id to a continuous index (starting from 1)
        self.cat_ids_to_continuous = {}
        for i, cat_id in enumerate(sorted(self.categories.keys())):
            self.cat_ids_to_continuous[cat_id] = i + 1
    
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
        img, anns = super(CocoDataset, self).__getitem__(index)
        
        # Get image id
        img_id = self.ids[index]
        
        # Initialize target dictionary
        target = {
            'boxes': [],
            'labels': [],
            'image_id': torch.tensor([img_id])
        }
        
        # Process annotations
        for ann in anns:
            # Skip annotations without bbox or with zero area
            if 'bbox' not in ann or ann['area'] <= 0:
                continue
                
            # Get bbox
            bbox = ann['bbox']
            
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            # Add bbox and label to target
            target['boxes'].append(bbox)
            target['labels'].append(self.cat_ids_to_continuous[ann['category_id']])
        
        # Skip images without annotations
        if len(target['boxes']) == 0:
            # Return a dummy target if no valid annotations
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

def get_coco_dataloader(root_dir, ann_file, batch_size, train=True, subset=False, num_workers=4):
    """
    Create COCO dataloader
    
    Args:
        root_dir: Directory with images
        ann_file: Path to annotations file
        batch_size: Batch size
        train: Whether to create training or validation dataloader
        subset: Whether to use a subset of categories
        num_workers: Number of workers for dataloader
        
    Returns:
        dataloader: COCO dataloader
    """
    # Define transforms
    if train:
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
    dataset = CocoDataset(
        root=root_dir,
        ann_file=ann_file,
        transform=transform,
        subset=subset
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader 