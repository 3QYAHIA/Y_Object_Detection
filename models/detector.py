import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class BackboneWithFPN(nn.Module):
    """
    Custom backbone with out_channels attribute for FasterRCNN
    """
    def __init__(self, backbone, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.backbone(x)
        return {'0': x}  # Return feature map as a dict

def get_faster_rcnn_model(num_classes, backbone='resnet50', pretrained=True, trainable_backbone_layers=3):
    """
    Create a Faster R-CNN model with the specified backbone
    
    Args:
        num_classes: Number of classes (including background)
        backbone: Backbone network ('resnet50' or 'mobilenet_v2')
        pretrained: Whether to use pretrained weights
        trainable_backbone_layers: Number of trainable layers in the backbone
        
    Returns:
        model: Faster R-CNN model with the specified backbone
    """
    
    if backbone == 'resnet50':
        # Load pre-trained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            backbone_net = resnet50(weights=weights)
        else:
            backbone_net = resnet50(weights=None)
            
        # Use only the feature extraction layers
        backbone_layers = nn.Sequential(*list(backbone_net.children())[:-2])
        
        # Set the output channels for the backbone
        backbone_out_channels = 2048
        
        # Create custom backbone with out_channels
        backbone = BackboneWithFPN(backbone_layers, backbone_out_channels)
        
    elif backbone == 'mobilenet_v2':
        # Load pre-trained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            backbone_net = mobilenet_v2(weights=weights)
        else:
            backbone_net = mobilenet_v2(weights=None)
        
        # Use the features part of MobileNetV2
        backbone_layers = backbone_net.features
        
        # Set the output channels for the backbone
        backbone_out_channels = 1280
        
        # Create custom backbone with out_channels
        backbone = BackboneWithFPN(backbone_layers, backbone_out_channels)
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Freeze layers based on trainable_backbone_layers
    for i, param in enumerate(backbone.backbone.parameters()):
        if i < len(list(backbone.backbone.parameters())) - trainable_backbone_layers * 2:
            param.requires_grad = False
    
    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Create ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=1000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25
    )
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def get_model_info(model):
    """Get model information (parameters, memory usage)"""
    
    # Count number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'backbone': model.backbone.__class__.__name__
    }
    
    return info 