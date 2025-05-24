"""
Image analysis models for object segmentation and volume measurement.

This module provides handlers for both RGB and RGBD (RGB + Depth) models,
implementing preprocessing, inference, and postprocessing for image segmentation.
"""

import os
import random
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import onnxruntime

import config as cfg # Import the new config file
from utils.depth_handler import DepthHandler
from utils.pointcloud import PointCloudGenerator

class Config:
    """Configuration settings for model initialization and inference."""
    
    # Model parameters
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.7
    RETINA_MASKS = True
    
    # Device settings
    DEVICE = 'cpu'

    # Image dimensions
    WIDTH = 512
    HEIGHT = 512
    
    # Visualization settings
    SHOW_LABELS = False
    SHOW_CONF = False
    SHOW_BOXES = False

def get_onnx_session(model_path: str, device: str) -> onnxruntime.InferenceSession:
    """
    Initialize and return the ONNX runtime inference session.
    
    Args:
        model_path: Path to ONNX model file
        device: Target device ('cuda', 'cpu')
        
    Returns:
        Initialized ONNX InferenceSession
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        providers = []
        if device == 'cuda' and onnxruntime.get_device() == 'GPU':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        return session
    except Exception as e:
        raise RuntimeError(f"Error loading ONNX model: {e}")

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
            model_path: Path to model weights file (.pt or .onnx). Will be converted to .onnx.
            logger: Logger instance for recording events
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.logger = logger # Assign logger at the beginning
        
        # Ensure we are attempting to load an .onnx model
        if model_path.endswith('.pt'):
            self.model_path = model_path.replace('.pt', '.onnx')
            self.logger.info(f"Original model path was .pt, changed to .onnx: {self.model_path}")
        else:
            self.model_path = model_path
            
        self.device = cfg.DEVICE 
        
        # Set random seed for reproducibility
        np.random.seed(40)
        random.seed(40)
        
        try:
            if not os.path.exists(self.model_path):
                # Fallback: if .onnx not found, maybe .pt exists and conversion script hasn't run for it
                pt_equivalent = self.model_path.replace('.onnx', '.pt')
                if os.path.exists(pt_equivalent):
                    self.logger.warning(f"ONNX model {self.model_path} not found, but .pt {pt_equivalent} exists. Run conversion script.")
                    raise FileNotFoundError(f'ONNX Model file not found: {self.model_path}. Corresponding .pt found, please convert.')
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            self.logger.info(f'Loading ONNX model from {self.model_path} for device {self.device}')
            self.session = get_onnx_session(self.model_path, self.device) 
            self.logger.info(f'ONNX model loaded successfully from {self.model_path}')
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.logger.info(f"ONNX Model Input: {self.input_name}, Outputs: {self.output_names}")

        except Exception as e:
            self.logger.error(f'Failed to load ONNX model: {str(e)}')
            raise
    
    def _calculate_2d_volume(self, mask: np.ndarray) -> float:
        """
        Calculate volume using 2D pixel-based estimation.
        
        Args:
            mask: Binary mask array (H, W) with values 0 or 255
            
        Returns:
            Estimated volume
        """
        pixel_count = np.sum(mask > 0)
        return (4/3) * np.power(pixel_count / np.pi, 3/2) if pixel_count > 0 else 0.0
    
    def _calculate_volumes_for_masks(self, masks: List[np.ndarray]) -> List[float]:
        """
        Calculate volumes for a list of masks using 2D estimation.
        
        Args:
            masks: List of binary mask arrays
            
        Returns:
            List of calculated volumes
        """
        return [self._calculate_2d_volume(mask) for mask in masks]
    
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
            Raw prediction outputs from ONNX model (list of numpy arrays)
        """
        # Preprocess the image
        processed_input = self.preprocess(image_bytes) # Returns np.ndarray (H, W, C)
        
        if processed_input.ndim == 3 and processed_input.shape[2] in [3, 4]: # HWC
            processed_input = np.transpose(processed_input, (2, 0, 1))  # CHW
        
        if processed_input.ndim == 3: # CHW
            processed_input = np.expand_dims(processed_input, axis=0) # BCHW

        if processed_input.dtype == np.uint8:
            processed_input = processed_input.astype(np.float32) / 255.0
        elif processed_input.dtype != np.float32:
            processed_input = processed_input.astype(np.float32)
        
        try:
            ort_inputs = {self.input_name: processed_input}
            raw_predictions = self.session.run(self.output_names, ort_inputs)
            return raw_predictions
        except Exception as e:
            self.logger.error(f"ONNX inference failed: {e}")
            self.logger.error(f"Input shape: {processed_input.shape}, dtype: {processed_input.dtype}")
            self.logger.error(f"Expected input name: {self.input_name}, output names: {self.output_names}")
            raise

    def postprocess(self, prediction) -> Tuple[np.ndarray, List[float]]:
        """
        Process raw model predictions into final output format.
        This method is now a wrapper for the ONNX-specific postprocessing.
        
        Args:
            prediction: Raw prediction from ONNX model (list of numpy arrays [output0, output1])
        
        Returns:
            tuple: (stacked_img, volumes) where:
                - stacked_img: Processed masks stacked vertically in a single image
                - volumes: List of calculated volumes for each mask
        """
        if prediction is None or not isinstance(prediction, list) or len(prediction) < 2: 
            self.logger.warning(f"Prediction is None or not in expected format (list of 2 arrays). Got: {type(prediction)}")
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []
        
        output0 = prediction[0] 
        proto_masks = prediction[1]

        return self._postprocess_onnx(output0, proto_masks)

    def _postprocess_onnx(self, output0: np.ndarray, proto_masks: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Process raw ONNX model predictions (detections and prototype masks) 
        into final output format (stacked binary masks and volumes).
        
        Args:
            output0 (np.ndarray): The first output of the ONNX model (detections).
                                Expected shape after initial processing: (num_predictions, num_attributes)
                                where num_attributes = 4 (bbox) + num_classes + num_mask_coeffs.
            proto_masks (np.ndarray): The second output of the ONNX model (prototype masks).
                                 Expected shape after initial processing: (num_mask_coeffs, mask_height, mask_width)
                                 e.g. (32, 128, 128) if proto masks are 128x128
        
        Returns:
            tuple: (stacked_img, volumes) where:
                - stacked_img: Processed binary masks stacked vertically.
                - volumes: List of calculated volumes for each mask.
        """
        # Remove batch dimension (assuming batch_size=1 for inference)
        if output0.shape[0] == 1:
            output0 = np.squeeze(output0, axis=0) 
        if proto_masks.shape[0] == 1:
            proto_masks = np.squeeze(proto_masks, axis=0) 

        # Ensure output0 is (num_predictions, num_attrs) by transposing if necessary
        # Default Ultralytics export: (num_attrs, num_predictions), e.g. (37, 5376) for 1 class, 32 coeffs
        if output0.shape[1] > output0.shape[0] and output0.shape[0] == (proto_masks.shape[0] + 4 + 1): # A common case: num_attrs is ~37, num_preds is 1000s
             output0 = output0.T # (num_predictions, num_attrs)
        
        num_predictions = output0.shape[0]
        num_mask_coeffs = proto_masks.shape[0]
        
        total_attrs = output0.shape[1]
        # num_classes = total_attrs - 4 (bbox) - num_mask_coeffs
        num_classes = total_attrs - 4 - num_mask_coeffs
        
        if num_classes < 1: 
            self.logger.error(f"Calculated num_classes < 1 ({num_classes}). total_attrs: {total_attrs}, num_mask_coeffs: {num_mask_coeffs}. Defaulting to 1.")
            num_classes = 1 
            # Validate if defaulting makes sense for the total_attrs
            expected_attrs_for_1_class = 4 + 1 + num_mask_coeffs
            if total_attrs != expected_attrs_for_1_class:
                 self.logger.error(f"Attribute count {total_attrs} mismatch even after forcing num_classes=1 (expected {expected_attrs_for_1_class}). Aborting postprocess.")
                 return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []

        bboxes_xywh = output0[:, :4]
        scores_all_classes = output0[:, 4 : 4 + num_classes]
        mask_coeffs = output0[:, 4 + num_classes : 4 + num_classes + num_mask_coeffs]

        if mask_coeffs.shape[1] != num_mask_coeffs:
             self.logger.error(f"CRITICAL: Mismatch in mask coefficients columns. Expected {num_mask_coeffs}, got {mask_coeffs.shape[1]}. num_classes: {num_classes}, total_attrs: {total_attrs}")
             return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []

        if num_classes == 1:
            class_ids = np.zeros(num_predictions, dtype=int)
            confidences = scores_all_classes[:, 0]
        else:
            class_ids = np.argmax(scores_all_classes, axis=1)
            confidences = np.max(scores_all_classes, axis=1)

        keep = confidences >= cfg.YOLO_CONF_THRESHOLD
        
        bboxes_xywh_filtered = bboxes_xywh[keep]
        confidences_filtered = confidences[keep]
        class_ids_filtered = class_ids[keep]
        mask_coeffs_filtered = mask_coeffs[keep]

        if len(bboxes_xywh_filtered) == 0:
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []

        x1 = bboxes_xywh_filtered[:, 0] - bboxes_xywh_filtered[:, 2] / 2
        y1 = bboxes_xywh_filtered[:, 1] - bboxes_xywh_filtered[:, 3] / 2
        x2 = bboxes_xywh_filtered[:, 0] + bboxes_xywh_filtered[:, 2] / 2
        y2 = bboxes_xywh_filtered[:, 1] + bboxes_xywh_filtered[:, 3] / 2
        bboxes_xyxy_filtered = np.stack([x1, y1, x2, y2], axis=1)

        indices_after_nms = cv2.dnn.NMSBoxes(
            bboxes_xyxy_filtered.tolist(), 
            confidences_filtered.tolist(), 
            cfg.YOLO_CONF_THRESHOLD, 
            cfg.YOLO_IOU_THRESHOLD
        )
        
        if isinstance(indices_after_nms, np.ndarray):
            indices_after_nms = indices_after_nms.flatten()
        elif not isinstance(indices_after_nms, list): # Handle empty tuple from NMSBoxes if no boxes
            indices_after_nms = []

        if len(indices_after_nms) == 0:
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []

        final_bboxes_xyxy = bboxes_xyxy_filtered[indices_after_nms]
        final_mask_coeffs = mask_coeffs_filtered[indices_after_nms]

        generated_masks = []
        original_h, original_w = cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH
        proto_h, proto_w = proto_masks.shape[1], proto_masks.shape[2] # e.g. 160, 160

        for i in range(len(final_bboxes_xyxy)):
            coeffs = final_mask_coeffs[i] 
            bbox_xyxy = final_bboxes_xyxy[i]

            instance_mask_proto_sized = (coeffs @ proto_masks.reshape(num_mask_coeffs, -1)).reshape(proto_h, proto_w)
            instance_mask_proto_sized = 1 / (1 + np.exp(-instance_mask_proto_sized))

            upsampled_mask = cv2.resize(instance_mask_proto_sized, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            full_image_mask = np.zeros((original_h, original_w), dtype=np.float32)
            x1, y1, x2, y2 = bbox_xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)
            
            if x1 < x2 and y1 < y2: # Valid box
                mask_segment = upsampled_mask[y1:y2, x1:x2]
                full_image_mask[y1:y2, x1:x2] = mask_segment
            
            binary_mask_final = (full_image_mask > 0.5).astype(np.uint8) * 255
            generated_masks.append(binary_mask_final)

        if generated_masks:
            # Use the common volume calculation method
            volumes = self._calculate_volumes_for_masks(generated_masks)
            stacked_img = np.vstack(generated_masks)
            self.logger.info(f"ONNX Postprocessing successful. Found {len(volumes)} objects.")
            return stacked_img, volumes
        
        self.logger.info("ONNX Postprocessing complete. No masks generated or met criteria.")
        return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH), dtype=np.uint8), []

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
        self.logger.info("Using RGBD model")
        try:
            # Initialize depth estimation model
            self.logger.info(f'Initializing depth handler with model: {depth_model_path}')
            # Pass the depth model type from main config if needed, or let DepthHandler use its default
            self.depth_handler = DepthHandler(depth_model_path, logger, model_type=cfg.DEPTH_MODEL_TYPE)
            
            # Initialize point cloud generator for volume calculation
            self.pc_generator = PointCloudGenerator(
                focal_length_x=cfg.FOCAL_LENGTH_X,
                focal_length_y=cfg.FOCAL_LENGTH_Y,
                mode="pca"
            )
            
            # Store current RGB image and depth map for point cloud generation
            self.current_rgb_image = None
            self.current_depth_map = None
            
            self.logger.info(f'RGBD model initialized successfully with point cloud support')
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
        
        # Store RGB and depth for later point cloud generation
        self.current_rgb_image = rgb_img.copy()
        self.current_depth_map = depth_map.copy()
        
        # Combine RGB and depth into RGBD
        rgbd_img = create_rgbd_image(rgb_img, depth_map)
        
        return rgbd_img
    
    def _calculate_volumes_for_masks(self, masks: List[np.ndarray]) -> List[float]:
        """
        Calculate volumes for masks using point cloud-based method with 2D fallback.
        
        Args:
            masks: List of binary mask arrays
            
        Returns:
            List of calculated volumes
        """
        if self.current_rgb_image is not None and self.current_depth_map is not None:
            try:
                volumes = self.pc_generator.calculate_volumes_from_masks(
                    self.current_rgb_image,
                    self.current_depth_map,
                    masks
                )
                self.logger.info(f"RGBD: Calculated {len(volumes)} point cloud-based volumes")
                return volumes
            except Exception as e:
                self.logger.error(f"Point cloud volume calculation failed: {e}")
                self.logger.info("Falling back to 2D volume calculation")
        else:
            self.logger.warning("No RGB/depth data available for point cloud calculation, using 2D fallback")
        
        # Fallback to parent's 2D calculation
        return super()._calculate_volumes_for_masks(masks)

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
        self.logger.info("Using RGB model")
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
