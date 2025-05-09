import os
import torch
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None, subset=False):
        """
        COCO dataset for object detection
        
        Args:
            root_dir: Root directory for images
            ann_file: Path to annotation file
            transforms: Optional transformations
            subset: Whether to use the subset version
        """
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.transforms = transforms
        
        # Get all image ids
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # Get categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.num_classes = len(self.categories) + 1  # +1 for background
        
        # Create category id to continuous index mapping
        self.cat_ids_to_continuous = {
            cat_id: i + 1  # +1 because 0 is background
            for i, cat_id in enumerate(sorted(self.categories.keys()))
        }
        
        print(f"Loaded {len(self.img_ids)} images with {len(self.categories)} categories")
    
    def __getitem__(self, idx):
        # Load image
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            # Skip annotations with no segmentation or tiny area
            if len(ann['segmentation']) == 0 or ann['area'] < 1:
                continue
                
            # Get box in [x, y, width, height] format
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2] format
            box = [x, y, x + w, y + h]
            boxes.append(box)
            
            # Convert category id to continuous index
            cat_id = ann['category_id']
            label = self.cat_ids_to_continuous[cat_id]
            labels.append(label)
        
        # Apply transformations - Important to do this BEFORE converting to tensors
        if self.transforms is not None:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to tensors
        target = {}
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # No boxes for this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        
        return img, target
    
    def __len__(self):
        return len(self.img_ids)

    def get_img_info(self, idx):
        img_id = self.img_ids[idx]
        return self.coco.loadImgs(img_id)[0]


def get_transform(train):
    """
    Get transformations for training or validation
    """
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_coco_dataloader(root_dir, ann_file, batch_size=4, train=True, subset=False):
    """
    Create and return COCO dataloader
    """
    dataset = CocoDataset(
        root_dir=root_dir,
        ann_file=ann_file,
        transforms=get_transform(train),
        subset=subset
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """
    Custom collate function for variable size inputs
    """
    return tuple(zip(*batch)) 