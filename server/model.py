"""
Image analysis models for object segmentation and volume measurement.

This module provides handlers for both RGB and RGBD (RGB + Depth) models,
implementing preprocessing, inference, and postprocessing for image segmentation.
"""

import os
import random
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import config as cfg # Import the new config file
from utils.depth_handler import DepthHandler

class Config:
    """Configuration settings for model initialization and inference."""
    
    # Model parameters
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    RETINA_MASKS = True
    
    # Device settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Image dimensions
    WIDTH = 512
    HEIGHT = 512
    
    # Visualization settings
    SHOW_LABELS = False
    SHOW_CONF = False
    SHOW_BOXES = False

def get_model(model_path: str) -> YOLO:
    """
    Initialize and return the YOLO model for inference.
    
    Args:
        model_path: Path to model weights file
        
    Returns:
        Initialized YOLO model
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")

def create_rgbd_image(rgb_image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """
    Create a 4-channel RGBD image by combining RGB and depth.
    
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

class BaseModelHandler(ABC):
    """Base class for all model handlers with common functionality."""
    
    def __init__(self, model_path: str, logger):
        """
        Initialize the base model handler.

        Args:
            model_path: Path to model weights file (.pt)
            logger: Logger instance for recording events
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path
        self.device = cfg.DEVICE # Use from config
        self.logger = logger
        
        # Set random seed for reproducibility
        torch.manual_seed(40)
        np.random.seed(40)
        random.seed(40)
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            self.logger.info(f'Loading YOLO model from {self.model_path}')
            self.model = get_model(self.model_path)
            self.logger.info(f'YOLO model loaded successfully from {model_path}')
        except Exception as e:
            self.logger.error(f'Failed to load YOLO model: {str(e)}')
            raise
    
    @abstractmethod
    def preprocess(self, image_bytes: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input - to be implemented by subclasses.
        
        Args:
            image_bytes: Input image as bytes or numpy array
            
        Returns:
            Numpy array ready for model input
        """
        pass
    
    def predict(self, image_bytes: Union[bytes, np.ndarray]):
        """
        Run inference on an image and return predictions.
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            Prediction results from YOLO model
        """
        # Preprocess the image
        processed_input = self.preprocess(image_bytes)
        
        # Perform inference
        results = self.model.predict(
            source=processed_input,
            conf=cfg.YOLO_CONF_THRESHOLD, # Use from config
            iou=cfg.YOLO_IOU_THRESHOLD, # Use from config
            imgsz=(cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), # Use from config
            device=self.device,
            retina_masks=cfg.YOLO_RETINA_MASKS, # Use from config
            show_labels=cfg.YOLO_SHOW_LABELS, # Use from config
            show_conf=cfg.YOLO_SHOW_CONF, # Use from config
            show_boxes=cfg.YOLO_SHOW_BOXES, # Use from config
            save=False,
            verbose=False
        )
        return results[0] if results else None

    def postprocess(self, prediction) -> Tuple[np.ndarray, List[float]]:
        """
        Process raw model predictions into final output format.
        
        Args:
            prediction: Raw prediction from YOLO model
        
        Returns:
            tuple: (stacked_img, volumes) where:
                - stacked_img: Processed masks stacked vertically in a single image
                - volumes: List of calculated volumes for each mask
        """
        if prediction is None or not hasattr(prediction, 'masks') or prediction.masks is None:
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []  # Use from config
        
        masks = []
        volumes = []
        
        # Convert masks to numpy arrays
        if len(prediction.masks.data) > 0:
            masks_data = prediction.masks.data.cpu().numpy()
            
            # Process all masks
            for mask in masks_data:
                # Ensure the mask is binary with 255 for white
                binary_mask = (mask > 0).astype(np.uint8) * 255
                
                # Calculate volume for this mask
                pixel_count = np.sum(binary_mask > 0)
                volume = (4/3) * np.power(pixel_count / np.pi, 3/2)
                
                masks.append(binary_mask)
                volumes.append(float(volume))
                
            if masks:
                # Stack the masks vertically for frontend display
                stacked_img = np.vstack(masks)
                return stacked_img, volumes
        
        return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []  # Use from config

class RGBDModelHandler(BaseModelHandler):
    """Handler for RGBD model that combines RGB and depth information."""
    
    def __init__(self, model_path: str, depth_model_path: str, logger):
        """
        Initialize the RGBD YOLO model handler.
        
        Args:
            model_path: Path to RGBD YOLO model weights file (.pt)
            depth_model_path: Path to Depth Anything model weights
            logger: Logger instance for recording events
        """
        # Initialize the base class
        super().__init__(model_path, logger)
        
        try:
            # Initialize depth estimation model
            self.logger.info(f'Initializing depth handler with model: {depth_model_path}')
            # Pass the depth model type from main config if needed, or let DepthHandler use its default
            self.depth_handler = DepthHandler(depth_model_path, logger, model_type=cfg.DEPTH_MODEL_TYPE)
            self.logger.info(f'RGBD model initialized successfully')
        except Exception as e:
            self.logger.error(f'Failed to initialize depth model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes: Union[bytes, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model input - generates RGB and depth, combines to RGBD.
        
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
        if image.size != (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT): # Use from config
            image = image.resize((cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT)) # Use from config
        
        # Convert to numpy array
        rgb_img = np.array(image)
        
        # Generate depth map using depth handler
        depth_map = self.depth_handler.predict(image_bytes)
        depth_map = self.depth_handler.postprocess(depth_map)
        
        # Combine RGB and depth into RGBD
        rgbd_img = create_rgbd_image(rgb_img, depth_map)
        
        return rgbd_img

class ModelHandler(BaseModelHandler):
    """Handler for standard RGB-only YOLO model."""
    
    def __init__(self, model_path: str, logger):
        """
        Initialize the standard RGB model handler.
        
        Args:
            model_path: Path to RGB YOLO model weights file (.pt)
            logger: Logger instance for recording events
        """
        # Initialize the base class
        super().__init__(model_path, logger)

    def preprocess(self, image_bytes: Union[bytes, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            Numpy array ready for model input
        """
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        else:
            # Already a PIL Image
            image = image_bytes
            
        # Resize if needed
        if image.size != (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT): # Use from config
            image = image.resize((cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT)) # Use from config
            
        # Convert PIL to numpy array
        return np.array(image)
