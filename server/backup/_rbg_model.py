from io import BytesIO
import os
import numpy as np
import torch
import random
from PIL import Image
from ultralytics import YOLO
import cv2

class Config:
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


def get_model(model_path):
    """Initialize and return the YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")


class ModelHandler:
    def __init__(self, model_path, logger):
        
        self.model_path = model_path
        self.device = Config.DEVICE
        
        # Set random seed for reproducibility
        torch.manual_seed(40)
        np.random.seed(40)
        random.seed(40)
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            logger.info(f'Loading YOLO model from {self.model_path}')
            self.model = get_model(self.model_path)
            logger.info(f'YOLO model loaded successfully from {model_path}')
            
        except Exception as e:
            logger.error(f'Failed to load YOLO model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes):
        """Preprocess image for model input"""
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            # Already a PIL Image
            image = image_bytes
        
        # Resize if needed
        if image.size != (Config.WIDTH, Config.HEIGHT):
            image = image.resize((Config.WIDTH, Config.HEIGHT))
            
        return image

    def predict(self, image_bytes):
        """Run inference on an image and return predictions"""
        
        # Preprocess the image
        image = self.preprocess(image_bytes)
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Perform inference
        results = self.model.predict(
            source=img_array,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            imgsz=(Config.WIDTH, Config.HEIGHT),
            device=self.device,
            retina_masks=Config.RETINA_MASKS,
            show_labels=Config.SHOW_LABELS,
            show_conf=Config.SHOW_CONF,
            show_boxes=Config.SHOW_BOXES,
            save=False,
            verbose=False
        )
        return results[0] if results else None
            
    def postprocess(self, prediction):
        if prediction is None or not hasattr(prediction, 'masks') or prediction.masks is None:
            print("No masks found in prediction.")
            return np.zeros((512, 512), dtype=np.uint8)  # Return an empty mask
        
        masks = []
        
        if len(prediction.masks.data) > 0:
            masks_data = prediction.masks.data.cpu().numpy()
            
            # Process all masks
            for mask in masks_data:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                masks.append(binary_mask)
                    
            if masks:
                stacked_img = np.vstack(masks)
                return stacked_img
        
        return np.zeros((512, 512), dtype=np.uint8)  # Return an empty mask
