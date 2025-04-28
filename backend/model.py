import os
import torch
import numpy as np
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import logging
from io import BytesIO  # Add this import for handling binary data

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # Model parameters
    MASK_THRESHOLD = 0.5
    BOX_DETECTIONS_PER_IMG = 250
    MIN_SCORE = 0.59
    
    # Other settings
    NORMALIZE = False
    DEVICE = torch.device('cpu')  # Using CPU as specified
    
    # ImageNet mean and std for normalization
    RESNET_MEAN = (0.485, 0.456, 0.406)
    RESNET_STD = (0.229, 0.224, 0.225)

    # Image dimensions
    WIDTH = 512
    HEIGHT = 512

def get_model():
    """Initialize and return the MaskR-CNN model"""
    # 2 classes: background (0) and fragment (1)
    NUM_CLASSES = 2
    
    if Config.NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False,  # We'll load trained weights
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
            image_mean=Config.RESNET_MEAN, 
            image_std=Config.RESNET_STD
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False,  # We'll load trained weights
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG
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

def remove_overlapping_pixels(mask, other_masks):
    """Remove pixels that overlap with other masks"""
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

class ModelHandler:
    def __init__(self, model_path):
        """Initialize the model handler with path to model weights"""
        self.model_path = model_path
        self.model = None
        self.device = Config.DEVICE
        self.initialized = False
        
    def load(self):
        """Load the model from disk"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = get_model()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.initialized = True
            logger.info("Model loaded successfully")
        except Exception as e:
            self.initialized = False
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess(self, image_bytes):
        """Preprocess image for model input"""
        if isinstance(image_bytes, bytes):
            # Convert bytes to PIL Image using BytesIO
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            # Already a PIL Image
            image = image_bytes
        
        # Resize if needed
        if image.size != (Config.WIDTH, Config.HEIGHT):
            image = image.resize((Config.WIDTH, Config.HEIGHT))
            
        # Convert to tensor
        img_tensor = F.to_tensor(image)
        
        # Normalize if needed
        if Config.NORMALIZE:
            img_tensor = F.normalize(img_tensor, Config.RESNET_MEAN, Config.RESNET_STD)
            
        return img_tensor
    
    def predict(self, image_bytes):
        """Run inference on an image and return predictions"""
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load() first.")
        
        try:
            # Preprocess the image
            img_tensor = self.preprocess(image_bytes)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model([img_tensor.to(self.device)])[0]
                
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
            
    def postprocess(self, predictions):
        """Process raw model predictions into final output format"""
        result = {
            'boxes': [],
            'scores': [],
            'labels': [],
            'masks': []
        }
        
        # Get image dimensions from the first mask
        if len(predictions['masks']) > 0:
            h, w = predictions['masks'][0].shape[1:]
        else:
            return result
        
        # Create a single combined mask for visualization
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        previous_masks = []
        
        for i, mask_tensor in enumerate(predictions['masks']):
            score = predictions['scores'][i].cpu().item()
            # Filter by confidence threshold
            if score < Config.MIN_SCORE:
                continue
                
            # Convert mask tensor to numpy array
            mask_np = mask_tensor.cpu().numpy()[0]
            binary_mask = (mask_np > Config.MASK_THRESHOLD).astype(np.uint8)
            
            # Remove overlapping pixels
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)
            
            # Add to combined mask
            combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
            
            # Get bounding box
            box = predictions['boxes'][i].cpu().numpy()
            label = predictions['labels'][i].cpu().item()
            
            # Add to result
            result['boxes'].append(box.tolist())
            result['scores'].append(score)
            result['labels'].append(label)
            result['masks'].append(binary_mask.tolist())  # Convert to list for JSON serialization
        
        result['combined_mask'] = combined_mask.tolist()  # Add combined mask
        return result
