import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# Configuration
CONFIG = {
    'input_path': '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images',
    'output_path': '/kaggle/working/depth_maps',
    'model_type': 'vitb',  # One of 'vits', 'vitb', 'vitl', 'vitg'
    'checkpoint_path': 'checkpoints/depth_anything_v2_{}.pth',
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'sanity_check_count': 3,  # Number of random images to check before full processing
}

# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_model(model_type):
    """Load the depth estimation model"""
    print(f"Loading {model_type} model...")
    model = DepthAnythingV2(**MODEL_CONFIGS[model_type])
    checkpoint_path = CONFIG['checkpoint_path'].format(model_type)
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model = model.to(CONFIG['device']).eval()
        print(f"Model {model_type} loaded successfully on {CONFIG['device']}")
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(model, image_path):
    """Process a single image and return the depth map"""
    try:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise FileNotFoundError(f"Image not found or could not be loaded from {image_path}")
            
        with torch.no_grad():
            depth = model.infer_image(raw_img)
        
        return depth, cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def save_depth_map(depth_map, output_path):
    """Save depth map as an image file"""
    if depth_map is None:
        return False
        
    # Normalize to 0-255 range
    depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
    
    try:
        cv2.imwrite(output_path, depth_normalized)
        return True
    except Exception as e:
        print(f"Error saving depth map to {output_path}: {e}")
        return False

def visualize_results(images_and_depths, title="Depth Map Visualization"):
    """Visualize images and their depth maps for sanity checking"""
    n = len(images_and_depths)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5*n))
    
    if n == 1:  # Handle the case when there's only one image
        axes = axes.reshape(1, -1)
    
    for i, (rgb_img, depth) in enumerate(images_and_depths):
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        
        # Display original image
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Display depth map
        depth_viz = axes[i, 1].imshow(depth_normalized, cmap='plasma')
        axes[i, 1].set_title("Depth Map")
        axes[i, 1].axis('off')
        
        # Add colorbar
        plt.colorbar(depth_viz, ax=axes[i, 1], shrink=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    print(f"Using device: {CONFIG['device']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['output_path'], exist_ok=True)
    
    # Load model
    model = load_model(CONFIG['model_type'])
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Get list of image files
    image_files = [f for f in os.listdir(CONFIG['input_path']) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images in input directory")
    
    # Select random images for sanity check
    random_images = random.sample(image_files, min(CONFIG['sanity_check_count'], len(image_files)))
    sanity_results = []
    
    print("\nPerforming sanity check on random images...")
    for img_file in random_images:
        img_path = os.path.join(CONFIG['input_path'], img_file)
        print(f"Processing {img_file} for sanity check...")
        depth, rgb_img = process_image(model, img_path)
        if depth is not None:
            sanity_results.append((rgb_img, depth))
    
    # Visualize sanity check results
    if sanity_results:
        visualize_results(sanity_results, "Sanity Check - Depth Maps")
    else:
        print("Sanity check failed. No images could be processed.")
        return
    
    # Process all images with progress bar
    print("\nProcessing all images...")
    success_count = 0
    
    for img_file in tqdm(image_files, desc="Generating depth maps"):
        img_path = os.path.join(CONFIG['input_path'], img_file)
        out_path = os.path.join(CONFIG['output_path'], img_file.replace('.jpg', '_depth.png'))
        
        depth, _ = process_image(model, img_path)
        if depth is not None:
            if save_depth_map(depth, out_path):
                success_count += 1
    
    print(f"\nDone! Successfully processed {success_count} out of {len(image_files)} images.")
    print(f"Depth maps saved to {CONFIG['output_path']}")

if __name__ == "__main__":
    main()
