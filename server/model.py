import os
import numpy as np
import torch
import random
from PIL import Image
from ultralytics import YOLO
import cv2
from io import BytesIO

# Import the depth handler
from depth_handler import DepthHandler

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
    """Initialize and return the YOLO model for RGBD inference"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")


def create_rgbd_image(rgb_image, depth_map):
    """Create a 4-channel RGBD image by stacking RGB and depth
    
    Args:
        rgb_image: RGB image as numpy array (H, W, 3)
        depth_map: Depth map as numpy array (H, W)
        
    Returns:
        RGBD image as numpy array (H, W, 4)
    """
    # Ensure images are the same size
    if rgb_image.shape[:2] != depth_map.shape[:2]:
        # Resize depth to match RGB
        depth_map = cv2.resize(
            depth_map,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Normalize depth map to 0-255 if it's not already
    if depth_map.dtype != np.uint8:
        min_val = depth_map.min()
        max_val = depth_map.max()
        if max_val > min_val:
            depth_map = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            depth_map = np.zeros_like(depth_map, dtype=np.uint8)
    
    # Create 4-channel RGBD image
    rgbd_image = np.dstack((rgb_image, depth_map))
    
    return rgbd_image


class RGBDModelHandler:
    def __init__(self, model_path, depth_model_path, logger):
        """Initialize the RGBD YOLO model handler
        
        Args:
            model_path: Path to RGBD YOLO model weights file (.pt)
            depth_model_path: Path to Depth Anything model weights
            logger: Logger instance for recording events
        """
        self.model_path = model_path
        self.device = Config.DEVICE
        
        # Set random seed for reproducibility
        torch.manual_seed(40)
        np.random.seed(40)
        random.seed(40)
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'RGBD model file not found: {self.model_path}')
            
            logger.info(f'Loading RGBD YOLO model from {self.model_path}')
            self.model = get_model(self.model_path)
            
            # Initialize depth estimation model
            logger.info(f'Initializing depth handler with model: {depth_model_path}')
            self.depth_handler = DepthHandler(depth_model_path, logger)
            
            logger.info(f'RGBD model initialized successfully')
            
        except Exception as e:
            logger.error(f'Failed to load RGBD model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes):
        """Preprocess image for model input - generates RGB and depth, combines to RGBD
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            RGBD image as numpy array
        """
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            # Already a PIL Image
            image = image_bytes
        
        # Resize if needed
        if image.size != (Config.WIDTH, Config.HEIGHT):
            image = image.resize((Config.WIDTH, Config.HEIGHT))
        
        # Convert to numpy array
        rgb_img = np.array(image)
        
        # Generate depth map using depth handler
        depth_map = self.depth_handler.predict(image_bytes)
        depth_map = self.depth_handler.postprocess(depth_map)
        
        # Combine RGB and depth into RGBD
        rgbd_img = create_rgbd_image(rgb_img, depth_map)
        
        return rgbd_img

    def predict(self, image_bytes):
        """Run inference on an image and return predictions
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            Prediction results
        """
        # Preprocess the image to get RGBD
        rgbd_img = self.preprocess(image_bytes)
        
        # Perform inference
        results = self.model.predict(
            source=rgbd_img,
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
        """Process raw model predictions into final output format
        
        Args:
            prediction: Raw prediction from YOLO model
        
        Returns:
            Processed masks stacked vertically in a single image
        """
        if prediction is None or not hasattr(prediction, 'masks') or prediction.masks is None:
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8)  # Return an empty mask
        
        masks = []
        
        # Convert masks to numpy arrays
        if len(prediction.masks.data) > 0:
            masks_data = prediction.masks.data.cpu().numpy()
            
            # Process all masks
            for mask in masks_data:
                # Ensure the mask is binary with 255 for white
                binary_mask = (mask > 0).astype(np.uint8) * 255
                masks.append(binary_mask)
                
            if masks:
                # Stack the masks vertically for frontend display
                stacked_img = np.vstack(masks)
                return stacked_img
        
        return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8)  # Return an empty mask
