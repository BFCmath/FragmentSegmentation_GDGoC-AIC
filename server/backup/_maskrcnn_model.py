from io import BytesIO
import os
import random

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO
import cv2


class MaskRCNNConfig:
    # Model parameters
    MASK_THRESHOLD = 0.5
    BOX_DETECTIONS_PER_IMG = 250
    MIN_SCORE = 0.59
    
    # Other settings
    DEVICE = torch.device('cpu')  # Using CPU as specified

    # Image dimensions
    WIDTH = 512
    HEIGHT = 512


class YOLOConfig:
    # Model parameters
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    RETINA_MASKS = True
    
    # Other settings
    DEVICE = 'cpu'  # Using CPU as specified

    # Image dimensions
    WIDTH = 512
    HEIGHT = 512
    
    # Visualization settings
    SHOW_LABELS = False
    SHOW_CONF = False
    SHOW_BOXES = False


def get_maskrcnn_model():
    """Initialize and return the MaskR-CNN model"""
    # 2 classes: background (0) and fragment (1)
    NUM_CLASSES = 2
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        box_detections_per_img=MaskRCNNConfig.BOX_DETECTIONS_PER_IMG
    )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # And replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    
    return model


def get_yolo_model(model_path):
    """Initialize and return the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")


class ModelHandler:
    def __init__(self, model_path, logger, model_type='maskrcnn'):
        """Initialize the model handler with path to model weights
        
        Args:
            model_path: Path to model weights file
            logger: Logger instance
            model_type: 'maskrcnn' or 'yolo'
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        
        # Set random seed for reproducibility
        torch.manual_seed(40)
        np.random.seed(40)
        random.seed(40)
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            if self.model_type == 'maskrcnn':
                self.device = MaskRCNNConfig.DEVICE
                logger.info(f'Loading MaskRCNN model from {self.model_path}')
                self.model = get_maskrcnn_model()
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
            elif self.model_type == 'yolo':
                self.device = YOLOConfig.DEVICE
                logger.info(f'Loading YOLO model from {self.model_path}')
                self.model = get_yolo_model(self.model_path)
            else:
                raise ValueError(f'Unsupported model type: {model_type}. Use "maskrcnn" or "yolo"')
                
            logger.info(f'{self.model_type.upper()} model loaded successfully from {model_path}')
        except Exception as e:
            logger.error(f'Failed to load {self.model_type} model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes):
        """Preprocess image for model input"""
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            # Already a PIL Image
            image = image_bytes
        
        if self.model_type == 'maskrcnn':
            # Resize if needed
            if image.size != (MaskRCNNConfig.WIDTH, MaskRCNNConfig.HEIGHT):
                image = image.resize((MaskRCNNConfig.WIDTH, MaskRCNNConfig.HEIGHT))
                
            # Convert to tensor
            img_tensor = F.to_tensor(image)
            return img_tensor
        else:  # yolo
            # Resize if needed
            if image.size != (YOLOConfig.WIDTH, YOLOConfig.HEIGHT):
                image = image.resize((YOLOConfig.WIDTH, YOLOConfig.HEIGHT))
                
            return image

    def predict(self, image_bytes):
        """Run inference on an image and return predictions"""
        # Preprocess the image
        if self.model_type == 'maskrcnn':
            img_tensor = self.preprocess(image_bytes)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model([img_tensor.to(self.device)])[0]
                
            return predictions
        else:  # yolo
            image = self.preprocess(image_bytes)
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Perform inference
            results = self.model.predict(
                source=img_array,
                conf=YOLOConfig.CONF_THRESHOLD,
                iou=YOLOConfig.IOU_THRESHOLD,
                imgsz=(YOLOConfig.WIDTH, YOLOConfig.HEIGHT),
                device=self.device,
                retina_masks=YOLOConfig.RETINA_MASKS,
                show_labels=YOLOConfig.SHOW_LABELS,
                show_conf=YOLOConfig.SHOW_CONF,
                show_boxes=YOLOConfig.SHOW_BOXES,
                save=False,
                verbose=False
            )
            # write to log.txt
            with open('log.txt', 'a') as f:
                f.write(f'{results[0].boxes}\n')
                f.write(f'{results[0].masks}\n')
            return results[0] if results else None
            
    def postprocess(self, predictions):
        """Process raw model predictions into final output format"""
        if self.model_type == 'maskrcnn':
            masks = []
            
            previous_masks = None
            
            for i, mask_tensor in enumerate(predictions['masks']):
                score = predictions['scores'][i].cpu().item()
                # Filter by confidence threshold
                if score < MaskRCNNConfig.MIN_SCORE:
                    continue
                    
                # Convert mask tensor to numpy array
                mask_np = mask_tensor.cpu().numpy()[0]
                binary_mask = mask_np >= MaskRCNNConfig.MASK_THRESHOLD
                
                # Remove overlapping pixels
                if previous_masks is None:
                    previous_masks = binary_mask.copy()
                else:
                    binary_mask[binary_mask & previous_masks] = 0
                    previous_masks |= binary_mask

                masks.append(binary_mask)
                
            if masks:
                return np.vstack(masks)
            return np.array([])
        else:  # yolo
            prediction = predictions
            if prediction is None or not hasattr(prediction, 'masks') or prediction.masks is None:
                return np.zeros((0, YOLOConfig.HEIGHT, YOLOConfig.WIDTH), dtype=np.uint8)  # Empty 3D array
            
            masks = []
            
            # Convert masks to numpy arrays
            if len(prediction.masks.data) > 0:
                masks_data = prediction.masks.data.cpu().numpy()
                
                # Stack all masks
                for mask in masks_data:
                    # Ensure the mask is binary
                    binary_mask = (mask > 0).astype(np.uint8)
                    masks.append(binary_mask)
                    
                if masks:
                    # For consistency with MaskRCNN, return a 2D array with masks stacked vertically
                    return np.vstack(masks)
                
            return np.array([])
