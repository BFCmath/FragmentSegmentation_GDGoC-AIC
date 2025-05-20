import os
import cv2
import torch
import numpy as np
from io import BytesIO
from PIL import Image

from utils.depth_anything_v2.dpt import DepthAnythingV2

class Config:
    MODEL_TYPE = 'vits'  # Small model (options: vits, vitb, vitl, vitg)
    
    # Other settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Image dimensions
    WIDTH = 512
    HEIGHT = 512
    
    # Model configurations
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }


def get_model(model_path, model_type='vits'):
    """Initialize and return the Depth Anything V2 model
    
    Args:
        model_path: Path to model weights file
        model_type: Model type (vits, vitb, vitl, vitg)
        
    Returns:
        Initialized model ready for inference
    """
    try:
        model = DepthAnythingV2(**Config.MODEL_CONFIGS[model_type])
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(Config.DEVICE).eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading Depth Anything model: {e}")


class DepthHandler:
    def __init__(self, model_path, logger, model_type='vits'):
        """Initialize the Depth Anything model handler
        
        Args:
            model_path: Path to Depth Anything model weights file (.pth)
            logger: Logger instance for recording events
            model_type: Type of model (vits, vitb, vitl, vitg)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = Config.DEVICE
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            logger.info(f'Loading Depth Anything model from {self.model_path}')
            self.model = get_model(self.model_path, self.model_type)
            logger.info(f'Depth Anything model loaded successfully')
            
        except Exception as e:
            logger.error(f'Failed to load Depth Anything model: {str(e)}')
            raise
    
    def preprocess(self, image_bytes):
        """Preprocess image for model input
        
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
    
    def predict(self, image_bytes):
        """Run depth estimation on an image and return depth map
        
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
            
    def postprocess(self, depth_map, normalize=True):
        """Process depth map into a visualizable format
        
        Args:
            depth_map: Depth map from model
            normalize: Whether to normalize to 0-255 range
            
        Returns:
            Processed depth map ready for visualization
        """
        if depth_map is None:
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8)
        
        if normalize:
            # Normalize to 0-255 range
            min_val = depth_map.min()
            max_val = depth_map.max()
            if max_val > min_val:
                depth_map = ((depth_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_map = np.zeros_like(depth_map, dtype=np.uint8)
        
        return depth_map