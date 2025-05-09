#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile
import json
import shutil
from tqdm import tqdm

def download_url(url, destination):
    """
    Downloads a file with a progress bar
    """
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, 
                                   reporthook=t.update_to)

def main():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    coco_dir = os.path.join(data_dir, "coco")
    
    os.makedirs(coco_dir, exist_ok=True)
    
    # URLs for COCO dataset 2017 - only download annotations and validation images (smaller)
    urls = {
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    # Download files
    for name, url in urls.items():
        filename = os.path.join(coco_dir, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {name}...")
            download_url(url, filename)
            
            # Extract the file
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
    
    print("COCO dataset (validation images only) downloaded and extracted successfully.")
    
    # Create a tiny subset of the dataset
    print("Creating a tiny subset of the COCO dataset (5 classes, 300 images)...")
    create_coco_tiny_subset(coco_dir)

def create_coco_tiny_subset(coco_dir, num_classes=5, max_images=300):
    """Creates a tiny subset of COCO with very few classes and images"""
    
    # Load the validation annotations (smaller than train)
    ann_file = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Get the most common classes
    class_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in class_counts:
            class_counts[cat_id] = 0
        class_counts[cat_id] += 1
    
    # Sort categories by frequency
    sorted_cats = sorted([(k, v) for k, v in class_counts.items()], 
                         key=lambda x: x[1], reverse=True)
    
    # Select top N categories
    selected_cats = [cat_id for cat_id, _ in sorted_cats[:num_classes]]
    
    # Create a category mapping
    old_to_new_cat = {}
    new_cats = []
    for i, cat_id in enumerate(selected_cats):
        for cat in data['categories']:
            if cat['id'] == cat_id:
                old_to_new_cat[cat_id] = i + 1  # 1-indexed
                new_cat = cat.copy()
                new_cat['id'] = i + 1
                new_cats.append(new_cat)
                break
    
    # Filter annotations by selected categories
    selected_anns = []
    selected_img_ids = set()
    
    # First pass: collect image IDs that have our selected categories
    for ann in data['annotations']:
        if ann['category_id'] in selected_cats:
            if len(selected_img_ids) < max_images:
                selected_img_ids.add(ann['image_id'])
    
    # Second pass: collect annotations for selected images
    for ann in data['annotations']:
        if ann['image_id'] in selected_img_ids and ann['category_id'] in selected_cats:
            new_ann = ann.copy()
            new_ann['category_id'] = old_to_new_cat[ann['category_id']]
            selected_anns.append(new_ann)
    
    # Filter images
    selected_imgs = []
    for img in data['images']:
        if img['id'] in selected_img_ids:
            selected_imgs.append(img)
    
    # Create new annotation file
    new_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': selected_imgs,
        'annotations': selected_anns,
        'categories': new_cats
    }
    
    # Save the subset annotations
    subset_dir = os.path.join(coco_dir, "tiny_subset")
    os.makedirs(os.path.join(subset_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, "val2017"), exist_ok=True)
    
    with open(os.path.join(subset_dir, "annotations", "instances_val2017.json"), 'w') as f:
        json.dump(new_data, f)
    
    # Copy the selected images
    for img in tqdm(selected_imgs, desc="Copying images"):
        src = os.path.join(coco_dir, "val2017", img['file_name'])
        dst = os.path.join(subset_dir, "val2017", img['file_name'])
        shutil.copy(src, dst)
    
    # Print dataset statistics
    total_size_mb = sum(os.path.getsize(os.path.join(subset_dir, "val2017", f)) for f in os.listdir(os.path.join(subset_dir, "val2017"))) / (1024 * 1024)
    
    print(f"Created tiny subset with:")
    print(f"- {len(new_cats)} classes: {', '.join([cat['name'] for cat in new_cats])}")
    print(f"- {len(selected_imgs)} images")
    print(f"- {len(selected_anns)} annotations")
    print(f"- Total size: {total_size_mb:.2f} MB")

if __name__ == "__main__":
    main() 