"""
Depth estimation handler for generating depth maps from RGB images.

This module provides functionality to load and use the Depth Anything V2 model
for estimating depth maps from RGB images, which are then used in RGBD processing.
"""

import os
from io import BytesIO
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

import config as cfg
from utils.depth_anything_v2.dpt import DepthAnythingV2


def get_model(model_path: str, model_type: str = 'vits') -> DepthAnythingV2:
    """
    Initialize and return the Depth Anything V2 model.
    
    Args:
        model_path: Path to model weights file
        model_type: Model type (vits, vitb, vitl, vitg)
        
    Returns:
        Initialized model ready for inference
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        # Use MODEL_CONFIGS and DEVICE from the main config
        model = DepthAnythingV2(**cfg.DEPTH_MODEL_CONFIGS[model_type])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(cfg.DEVICE).eval() # Use DEVICE from main config
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading Depth Anything model: {e}")


class DepthHandler:
    """Handles depth estimation using Depth Anything V2 model."""
    
    def __init__(self, model_path: str, logger, model_type: str = cfg.DEPTH_MODEL_TYPE):
        """
        Initialize the Depth Anything model handler.
        
        Args:
            model_path: Path to Depth Anything model weights file (.pth)
            logger: Logger instance for recording events
            model_type: Type of model (e.g., 'vits', 'vitb'), defaults to cfg.DEPTH_MODEL_TYPE
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = cfg.DEVICE # Use DEVICE from main config
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            logger.info(f'Loading Depth Anything model from {self.model_path}')
            self.model = get_model(self.model_path, self.model_type)
            logger.info(f'Depth Anything model loaded successfully')
            
        except Exception as e:
            logger.error(f'Failed to load Depth Anything model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            OpenCV image in BGR format (model's expected input)
        """
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            # Convert PIL to numpy array (RGB)
            rgb_img = np.array(pil_image)
            # Convert RGB to BGR (OpenCV format)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        else:
            # Assume it's already a numpy array
            bgr_img = image_bytes
            
        return bgr_img
    
    def predict(self, image_bytes: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Run depth estimation on an image and return depth map.
        
        Args:
            image_bytes: Input image as bytes or PIL Image
            
        Returns:
            Depth map as numpy array
        """
        # Preprocess the image
        bgr_img = self.preprocess(image_bytes)
        
        # Perform inference
        with torch.no_grad():
            depth_map = self.model.infer_image(bgr_img)
        
        return depth_map
            
    def postprocess(self, depth_map: Optional[np.ndarray], normalize: bool = True) -> np.ndarray:
        """
        Process depth map into a visualizable format.
        
        Args:
            depth_map: Depth map from model
            normalize: Whether to normalize to 0-255 range
            
        Returns:
            Processed depth map ready for visualization
        """
        if depth_map is None:
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8) # Use from config
        
        if normalize:
            # Normalize to 0-255 range
            min_val = depth_map.min()
            max_val = depth_map.max()
            if max_val > min_val:
                depth_map = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_map = np.zeros_like(depth_map, dtype=np.uint8)
        
        return depth_map