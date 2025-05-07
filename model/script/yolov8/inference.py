import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2
import time
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)

class Config:
    def __init__(self):
        # Model settings
        self.model_path = '/kaggle/working/best.pt'  # Path to trained model
        
        # Inference settings
        self.conf_threshold = 0.25  # Confidence threshold
        self.iou_threshold = 0.7    # IoU threshold for NMS
        self.device = 'cpu'         # Force CPU usage
        
        # Single image source and output path
        self.image_path = '/kaggle/input/fst-depth-map/vits/001_depth.png'  # Path to single test image
        self.output_path = '/kaggle/working/pred'  # Path to save predictions
        
        # Visualization settings
        self.imgsz = 512            # Same size used during training
        self.hide_labels = True     # Hide labels
        self.hide_conf = True       # Hide confidences
        
        # Class information
        self.classes = ['fragment']  # Single class: fragment

def load_model(config):
    """Load the trained YOLOv8 model"""
    try:
        model = YOLO(config.model_path)
        print(f"Model loaded successfully from {config.model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_single_inference(model, config):
    """Run inference on a single image"""
    if not model:
        print("No model loaded. Cannot run inference.")
        return None
    
    # Ensure image path exists
    if not os.path.exists(config.image_path):
        print(f"Image not found at {config.image_path}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_path, exist_ok=True)
    
    # Run prediction with the model and time it
    print(f"Running inference on {config.image_path}")
    
    # Start timing
    start_time = time.time()
    
    results = model.predict(
        source=config.image_path,
        conf=config.conf_threshold,
        iou=config.iou_threshold,
        imgsz=config.imgsz,
        device=config.device,
        save=True,
        save_txt=False,  # No need to save text for single image
        project=config.output_path,
        name='predict',
        visualize=False,  # We'll do our own visualization
        hide_labels=True,
        hide_conf=True,
        boxes=False,      # No bounding boxes
        retina_masks=True # High-quality segmentation masks
    )
    
    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    
    return results[0], inference_time  # Return only the first result and time

def visualize_single_result(result, img_path, inference_time, output_path):
    """Visualize single image prediction with side-by-side comparison"""
    if result is None:
        print("No result to visualize")
        return
    
    # Create figure with side-by-side layout
    plt.figure(figsize=(15, 6))
    
    # Show original image on the left
    plt.subplot(1, 2, 1)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Original Image: {Path(img_path).name}")
    plt.axis('off')
    
    # Show masked prediction on the right
    plt.subplot(1, 2, 2)
    
    # Create custom visualization with different colors for each instance
    if hasattr(result, 'masks') and result.masks is not None:
        # Read original image
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Get mask data
            masks = result.masks.data.cpu().numpy()
            
            # Create a blank image for the masks with same dimensions as original
            mask_img = np.zeros_like(orig_img)
            
            # Add each mask with a different color
            for j, mask in enumerate(masks):
                # Generate a random color for each instance
                color = np.array([random.randint(0, 255), 
                                 random.randint(0, 255), 
                                 random.randint(0, 255)])
                
                # Get mask dimensions
                mask_h, mask_w = mask.shape
                img_h, img_w = orig_img.shape[:2]
                
                # Reshape mask to match image dimensions
                bin_mask = mask.astype('uint8')
                if mask_h != img_h or mask_w != img_w:
                    bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Apply the mask with the random color
                mask_img[bin_mask > 0] = color
            
            # Blend the original image with the masks
            alpha = 0.5
            blended = cv2.addWeighted(orig_img, 1-alpha, mask_img, alpha, 0)
            plt.imshow(blended)
            plt.title(f"Prediction ({len(masks)} objects, {inference_time:.3f}s)")
        else:
            # If we can't load the original image, use the default plotting
            plt.imshow(result.plot(boxes=False, labels=False, conf=False))
            plt.title(f"Prediction ({inference_time:.3f}s)")
    else:
        # If there are no masks, use the default plotting
        plt.imshow(result.plot(boxes=False, labels=False, conf=False))
        plt.title(f"Prediction (No masks detected, {inference_time:.3f}s)")
    
    plt.axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_path, f"prediction_{Path(img_path).stem}.png")
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.show()

def main():
    """Main function"""
    # Initialize configuration
    config = Config()
    
    # Load model
    model = load_model(config)
    if not model:
        return
    
    # Run inference on single image
    result, inference_time = run_single_inference(model, config)
    
    # Visualize result
    if result:
        visualize_single_result(result, config.image_path, inference_time, config.output_path)
    
    print("Inference completed!")

if __name__ == '__main__':
    main()