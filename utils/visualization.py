import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from PIL import Image
import random
import time
import os
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def plot_image_with_boxes(img, boxes, labels=None, scores=None, class_names=None, figsize=(12, 12)):
    """
    Plot image with bounding boxes, labels and scores
    
    Args:
        img: Image tensor or array (C, H, W)
        boxes: Bounding boxes tensor (N, 4) in [x1, y1, x2, y2] format
        labels: Optional labels tensor (N)
        scores: Optional scores tensor (N)
        class_names: Optional list of class names
        figsize: Figure size (width, height)
    """
    # Convert tensor to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C) for matplotlib
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # Denormalize if image values are in [0, 1] range
    if img.max() <= 1.0:
        img = img * 255
    
    # Convert to uint8
    img = img.astype(np.uint8)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    
    # Convert boxes to numpy array if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    
    # Generate random colors for each class
    colors = []
    if class_names:
        for _ in range(len(class_names) + 1):  # +1 for background
            colors.append((random.random(), random.random(), random.random()))
    else:
        # If no class names, use a single color
        colors = [(1, 0, 0)]
    
    # Plot each box
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Create rectangle patch
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                               edgecolor=colors[labels[i] if labels is not None else 0],
                               facecolor='none')
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add label and score if available
        if labels is not None and scores is not None:
            label_idx = labels[i]
            score = scores[i]
            
            if class_names:
                label_text = f"{class_names[label_idx - 1]}: {score:.2f}" if label_idx > 0 else f"background: {score:.2f}"
            else:
                label_text = f"Class {label_idx}: {score:.2f}"
                
            ax.text(x1, y1, label_text, 
                   bbox=dict(facecolor=colors[label_idx if labels is not None else 0], alpha=0.5))
    
    plt.axis('off')
    return fig


def visualize_prediction(image, prediction, dataset, threshold=0.5, figsize=(12, 12)):
    """
    Visualize model predictions on an image
    
    Args:
        image: Image tensor (C, H, W)
        prediction: Model prediction dictionary with 'boxes', 'labels', 'scores'
        dataset: Dataset with category information
        threshold: Score threshold for displaying boxes
        figsize: Figure size
    """
    # Get boxes, labels and scores
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    
    # Get confidence mask
    mask = scores >= threshold
    
    # Filter boxes, labels and scores by confidence
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Get class names
    class_names = []
    
    # Create reverse mapping from continuous index to original category id
    for cat_id, idx in dataset.cat_ids_to_continuous.items():
        if idx <= len(dataset.categories):
            cat_name = dataset.categories[cat_id]
            # Ensure the list is large enough
            while len(class_names) < idx:
                class_names.append("unknown")
            class_names[idx - 1] = cat_name
    
    # Plot image with boxes
    return plot_image_with_boxes(image, boxes, labels, scores, class_names, figsize)


def save_detection_visualization(model, dataset, images, targets, output_dir, num_samples=5, threshold=0.5):
    """
    Run model on images and save visualizations
    
    Args:
        model: Detection model
        dataset: Dataset with category information
        images: List of images
        targets: List of targets
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        threshold: Score threshold for displaying boxes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Process at most num_samples images
    for i in range(min(num_samples, len(images))):
        # Get image
        image = images[i]
        
        # Run model
        with torch.no_grad():
            prediction = model([image.to(next(model.parameters()).device)])[0]
        
        # Move prediction to CPU for visualization
        for k, v in prediction.items():
            prediction[k] = v.cpu()
        
        # Visualize prediction
        fig = visualize_prediction(image, prediction, dataset, threshold)
        
        # Save figure
        fig.savefig(os.path.join(output_dir, f"detection_{i}.png"), bbox_inches='tight')
        plt.close(fig) 