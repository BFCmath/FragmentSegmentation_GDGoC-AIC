import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
import time # Import the time module

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Load the image once before the loop as it's the same for all models
image_path = '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images/001.jpg'
try:
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Image not found or could not be loaded from {image_path}")
    # Convert from BGR to RGB for proper display in matplotlib
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    print(f"Successfully loaded image: {image_path}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    # Exit or handle the error appropriately if the image can't be loaded
    # For this example, we'll just skip the loop if the image is not loaded
    rgb_img = None # Set to None to indicate loading failed

if rgb_img is not None:
    for encoder in ['vits', 'vitb', 'vitl']: # Added vits and vitb back as per your original model_configs
        print(f"\nProcessing with {encoder} model...")

        # Initialize and load the model
        model = DepthAnythingV2(**model_configs[encoder])
        try:
            model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
            model = model.to(DEVICE).eval()
            print(f"Model {encoder} loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found for {encoder} at checkpoints/depth_anything_v2_{encoder}.pth. Skipping this model.")
            continue # Skip to the next encoder if checkpoint is missing
        except Exception as e:
            print(f"An error occurred while loading model {encoder}: {e}")
            continue


        # Perform inference and time it
        # Ensure inference is done within torch.no_grad() for efficiency and to disable gradient calculation
        with torch.no_grad():
            start_time = time.perf_counter() # Start the timer
            depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
            end_time = time.perf_counter()   # Stop the timer

        duration = end_time - start_time
        print(f"Inference time for {encoder} model: {duration:.4f} seconds") # Print the duration

        # --- Visualization ---
        # Normalize depth for better visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"{encoder} Original Map", fontsize=16)
        axes[0].axis('off')

        # Depth map with colormap
        depth_viz = axes[1].imshow(depth_normalized, cmap='plasma')
        axes[1].set_title(f'{encoder} Depth Map', fontsize=16)
        axes[1].axis('off')

        # Add colorbar
        cbar = fig.colorbar(depth_viz, ax=axes[1], shrink=0.7)
        cbar.set_label('Depth Value (Normalized)', fontsize=12)

        plt.tight_layout()
        plt.show()

else:
    print("Skipping processing models due to image loading failure.")