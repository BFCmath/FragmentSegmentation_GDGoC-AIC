# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:11.532504Z","iopub.execute_input":"2025-04-25T08:21:11.532780Z","iopub.status.idle":"2025-04-25T08:21:19.824495Z","shell.execute_reply.started":"2025-04-25T08:21:11.532743Z","shell.execute_reply":"2025-04-25T08:21:19.823752Z"}}
# Import libraries
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import cv2
from pycocotools import mask as coco_mask
import pickle
import os.path as osp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Configuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:19.825174Z","iopub.execute_input":"2025-04-25T08:21:19.825519Z","iopub.status.idle":"2025-04-25T08:21:19.917680Z","shell.execute_reply.started":"2025-04-25T08:21:19.825498Z","shell.execute_reply":"2025-04-25T08:21:19.916489Z"}}
# Data paths
TRAIN_IMAGE_DIR = '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images'
TRAIN_MASK_DIR = '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/masks'
VAL_IMAGE_DIR = '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/val/images'

# Model & Training parameters
NAME_VERSION = "mask-rcnn-v1.0"
TARGET_SIZE = 512  # Image size for training and inference
BATCH_SIZE = 4     # Smaller batch size for Mask R-CNN which is more memory intensive
NUM_EPOCHS = [2, 6, 8]  # Three-stage training like in reference.py
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5  # L2 regularization
SEED = 42  # Random seed for reproducibility
NUM_WORKERS = min(8, os.cpu_count() or 1)  # Number of workers for data loading

# RPN configuration (from reference.py)
RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # Anchor box scales

# Caching configuration
USE_CACHE = True  # Whether to use instance extraction cache
EXTRACTION_CACHE_PATH = None  # Set to a path to use a specific cache location
FORCE_REFRESH = False  # Whether to force rebuild the cache

# Inference parameters
CONFIDENCE_THRESHOLD = 0.3  # Reduced from 0.5
VISUALIZATION_THRESHOLD = 0.7  # Higher threshold for visualization

# Output paths
OUTPUT_DIR = f'/kaggle/working/{NAME_VERSION}'
INSTANCE_SUBMISSION_NAME = 'instance_submission.csv'
BINARY_SUBMISSION_NAME = 'binary_submission.csv'

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [code] {"execution":{"iopub.status.busy":"2025-04-25T08:21:19.918777Z","iopub.execute_input":"2025-04-25T08:21:19.919183Z","iopub.status.idle":"2025-04-25T08:21:19.941637Z","shell.execute_reply.started":"2025-04-25T08:21:19.919140Z","shell.execute_reply":"2025-04-25T08:21:19.940886Z"}}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Centralized Instance Extraction

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:19.942495Z","iopub.execute_input":"2025-04-25T08:21:19.942791Z","iopub.status.idle":"2025-04-25T08:21:19.959414Z","shell.execute_reply.started":"2025-04-25T08:21:19.942760Z","shell.execute_reply":"2025-04-25T08:21:19.958610Z"}}
def extract_instances_from_mask(mask):
    """
    Extract individual instances from a mask.
    For instance segmentation tasks where:
    - Each fragment is a separate instance with a unique color ID
    - All instances belong to the same class (fragment)
    - Background is represented by 0
    """
    # Count unique values in mask
    unique_values = np.unique(mask)
    
    # Check if this is a multi-class instance mask (like the one shown in the image)
    if len(unique_values) > 2:  # Multi-class mask (each instance has a different value)
        unique_values = unique_values[unique_values > 0]  # Skip 0 (background)
        
        instances = []
        for value in unique_values:
            instance_mask = (mask == value).astype(np.uint8)
            
            # Find bounding box
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
                
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            y_min, y_max = y_indices[[0, -1]]
            x_min, x_max = x_indices[[0, -1]]
            
            # Ensure box has positive width and height (minimum 1 pixel)
            if x_min == x_max:
                x_max = min(x_min + 1, mask.shape[1] - 1)
            if y_min == y_max:
                y_max = min(y_min + 1, mask.shape[0] - 1)
            
            # Store instance info
            instances.append({
                "mask": instance_mask,
                "bbox": [x_min, y_min, x_max, y_max]
            })
    else:  # Binary mask - use connected components to separate instances
        # Find connected components to separate instances
        ret, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        instances = []
        for label_id in range(1, ret):  # Skip 0 (background)
            instance_mask = (labels == label_id).astype(np.uint8)
            
            # Find bounding box
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue
                
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            y_min, y_max = y_indices[[0, -1]]
            x_min, x_max = x_indices[[0, -1]]
            
            # Ensure box has positive width and height (minimum 1 pixel)
            if x_min == x_max:
                x_max = min(x_min + 1, mask.shape[1] - 1)
            if y_min == y_max:
                y_max = min(y_min + 1, mask.shape[0] - 1)
            
            # Store instance info
            instances.append({
                "mask": instance_mask,
                "bbox": [x_min, y_min, x_max, y_max]
            })
    
    return instances

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## RLE Preprocessing Pipeline

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:19.961578Z","iopub.execute_input":"2025-04-25T08:21:19.961778Z","iopub.status.idle":"2025-04-25T08:21:19.979546Z","shell.execute_reply.started":"2025-04-25T08:21:19.961761Z","shell.execute_reply":"2025-04-25T08:21:19.978889Z"}}
def preprocess_all_masks_to_rle(mask_dir, image_list, cache_path, target_size=TARGET_SIZE):
    """
    Preprocess all masks in the dataset to RLE format and cache them
    This is done once before training to avoid repeated extraction
    
    Args:
        mask_dir: Directory containing masks
        image_list: List of image files (used to find corresponding masks)
        cache_path: Path to save the RLE cache
        target_size: Size to resize masks to before conversion
        
    Returns:
        rle_cache: Dictionary mapping image names to RLEs and instance data
    """
    print(f"Preprocessing {len(image_list)} masks to RLE format...")
    rle_cache = {}
    
    for img_file in tqdm(image_list):
        mask_file = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}")
            continue
            
        # Load mask
        mask = np.array(Image.open(mask_path))
        
        # Extract and process instances
        instances = extract_instances_from_mask(mask)
        
        # Store both instance data and RLEs
        instance_masks = []
        instance_boxes = []
        instance_rles = []
        
        # Create a combined binary mask for all instances
        binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        
        for instance in instances:
            instance_mask = instance["mask"]
            instance_bbox = instance["bbox"]
            
            # Update binary mask
            binary_mask = np.logical_or(binary_mask, instance_mask).astype(np.uint8)
            
            # Convert mask to RLE
            instance_rle = mask_to_rle(instance_mask)
            
            # Store info
            instance_masks.append(instance_mask)
            instance_boxes.append(instance_bbox)
            instance_rles.append(instance_rle)
            
        # Convert binary mask to RLE
        binary_rle = mask_to_rle(binary_mask)
        
        # Store in cache
        rle_cache[img_file] = {
            'instance_masks': instance_masks,
            'instance_boxes': instance_boxes,
            'instance_rles': instance_rles,
            'binary_rle': binary_rle
        }
    
    # Save cache
    print(f"Saving RLE cache with {len(rle_cache)} entries to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(rle_cache, f)
    
    return rle_cache

# Function to load RLE from cached representation
def rle_to_mask(rle, shape):
    """
    Convert RLE back to binary mask
    
    Args:
        rle: Run-length encoded mask string
        shape: Output shape (height, width)
        
    Returns:
        Binary mask
    """
    if not rle:
        return np.zeros(shape, dtype=np.uint8)
        
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert from 1-indexed to 0-indexed
    ends = starts + lengths
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    
    return mask.reshape(shape, order='F')  # Reshape using Fortran order (column-major)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:19.980689Z","iopub.execute_input":"2025-04-25T08:21:19.980875Z","iopub.status.idle":"2025-04-25T08:21:22.291580Z","shell.execute_reply.started":"2025-04-25T08:21:19.980859Z","shell.execute_reply":"2025-04-25T08:21:22.290755Z"}}
# Set up cache directories and files
if EXTRACTION_CACHE_PATH is None:
    cache_dir = os.path.join(OUTPUT_DIR, 'cache')
    USE_PRECACHED_RLE = False
else:
    cache_dir = EXTRACTION_CACHE_PATH
    USE_PRECACHED_RLE = True  # Use pre-extracted RLEs

os.makedirs(cache_dir, exist_ok=True)

# Define cache file paths
rle_cache_file = os.path.join(cache_dir, 'all_masks_rle.pkl')

# Check if files exist in directories
if not os.path.exists(TRAIN_IMAGE_DIR):
    raise FileNotFoundError(f"Training image directory not found: {TRAIN_IMAGE_DIR}")
if not os.path.exists(TRAIN_MASK_DIR):
    raise FileNotFoundError(f"Training mask directory not found: {TRAIN_MASK_DIR}")
if not os.path.exists(VAL_IMAGE_DIR):
    raise FileNotFoundError(f"Validation image directory not found: {VAL_IMAGE_DIR}")

# Get all images first for potential extraction
all_images = [f for f in os.listdir(TRAIN_IMAGE_DIR) if f.endswith('.jpg')]
print(f"Found {len(all_images)} images in training directory")

# Verify mask files exist for training images
valid_images = []
for img_file in all_images:
    mask_file = os.path.splitext(img_file)[0] + '.png'
    if os.path.exists(os.path.join(TRAIN_MASK_DIR, mask_file)):
        valid_images.append(img_file)

if len(valid_images) < len(all_images):
    print(f"Warning: Only {len(valid_images)} of {len(all_images)} images have corresponding masks")

# Split data into train and validation
train_images, valid_images = train_test_split(
    valid_images, test_size=0.2, random_state=SEED
)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(valid_images)}")

# Global RLE cache
rle_cache = {}

# If extraction cache path is provided, preprocess all masks to RLE
if USE_PRECACHED_RLE:
    if os.path.exists(rle_cache_file) and not FORCE_REFRESH:
        print(f"Loading cached RLEs from {rle_cache_file}")
        try:
            with open(rle_cache_file, 'rb') as f:
                rle_cache = pickle.load(f)
            print(f"Successfully loaded {len(rle_cache)} RLE entries from cache")
        except Exception as e:
            print(f"Error loading RLE cache: {e}")
            rle_cache = {}
    
    if not rle_cache or FORCE_REFRESH:
        print("RLE cache not found or refresh forced. Preprocessing masks to RLE...")
        rle_cache = preprocess_all_masks_to_rle(
            TRAIN_MASK_DIR,
            valid_images,  # All valid images (train + validation)
            rle_cache_file,
            TARGET_SIZE
        )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Seed for Reproducibility

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.292488Z","iopub.execute_input":"2025-04-25T08:21:22.292766Z","iopub.status.idle":"2025-04-25T08:21:22.301954Z","shell.execute_reply.started":"2025-04-25T08:21:22.292744Z","shell.execute_reply":"2025-04-25T08:21:22.301176Z"}}
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Utility Functions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.302976Z","iopub.execute_input":"2025-04-25T08:21:22.303219Z","iopub.status.idle":"2025-04-25T08:21:22.326358Z","shell.execute_reply.started":"2025-04-25T08:21:22.303198Z","shell.execute_reply":"2025-04-25T08:21:22.325533Z"}}
# IoU (Jaccard Index) for evaluation as per competition metric
def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Function to save training history with IoU scores
def save_training_history(train_losses, val_losses, learning_rates, train_ious=None, val_ious=None, save_dir=OUTPUT_DIR):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics to CSV
    history = {
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rate': learning_rates
    }
    
    if train_ious is not None:
        history['train_iou'] = train_ious
    if val_ious is not None:
        history['val_iou'] = val_ious
        
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], train_losses, 'b-', label='Training Loss')
    plt.plot(history['epoch'], val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], learning_rates, 'g-')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    # Add IoU plots if available
    if train_ious is not None and val_ious is not None:
        plt.subplot(2, 2, 3)
        plt.plot(history['epoch'], train_ious, 'b-', label='Training IoU')
        plt.plot(history['epoch'], val_ious, 'r-', label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU Score')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

# RLE encoding function
def mask_to_rle(mask):
    """
    Convert a binary mask to run-length encoding (RLE)
    Following the competition's approach but with error handling.
    """
    # Handle empty masks
    if mask.sum() == 0:
        return ''
        
    # Flatten mask in 'C' order (row-major) as per competition
    pixels = mask.flatten()
    
    # Add padding at beginning and end
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find transitions (where pixel value changes)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # Handle edge case where the runs array has odd length
    if len(runs) % 2 != 0:
        print(f"Warning: Odd number of transitions ({len(runs)}) in mask. Adjusting...")
        # Make it even by adding or removing a transition
        if pixels[-1] == 1:
            # If the mask ends with 1, add an extra transition at the end
            runs = np.append(runs, len(pixels))
        else:
            # If mask ends with 0, remove the last transition if safe to do so
            if len(runs) > 1:
                runs = runs[:-1]
            else:
                # Empty result case
                return ''
    
    # Calculate run lengths as per competition method
    runs[1::2] -= runs[::2]
    
    # Convert to space-separated string
    rle = ' '.join(map(str, runs))
    return rle

# Visualization function for images with instances
def visualize_predictions(model, images, targets=None, device=DEVICE, num_images=3):
    """Visualize predictions from Mask R-CNN model"""
    model.eval()
    fig, axs = plt.subplots(num_images, 3 if targets else 2, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, image in enumerate(images[:num_images]):
            # Forward pass
            image_tensor = image.to(device)
            output = model([image_tensor])[0]
            
            # Plot original image
            img_np = image.permute(1, 2, 0).cpu().numpy()
            axs[i, 0].imshow(img_np)
            axs[i, 0].set_title("Original Image")
            axs[i, 0].axis('off')
            
            # Plot predictions
            axs[i, 1].imshow(img_np)
            axs[i, 1].set_title("Predictions")
            axs[i, 1].axis('off')
            
            # Draw predicted masks and bounding boxes
            for j, mask in enumerate(output['masks']):
                if output['scores'][j] > VISUALIZATION_THRESHOLD:  # Only show high confidence predictions
                    mask = mask.squeeze().cpu().numpy() > 0.5
                    color = np.random.rand(3)
                    masked_img = np.zeros_like(img_np)
                    for c in range(3):
                        masked_img[:, :, c] = np.where(mask, color[c], 0)
                    axs[i, 1].imshow(masked_img, alpha=0.5)
                    
                    # Draw bounding box
                    box = output['boxes'][j].cpu().numpy()
                    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                        linewidth=2, edgecolor=color, facecolor='none')
                    axs[i, 1].add_patch(rect)
            
            # Plot ground truth if available
            if targets:
                axs[i, 2].imshow(img_np)
                axs[i, 2].set_title("Ground Truth")
                axs[i, 2].axis('off')
                
                target = targets[i]
                for j, mask in enumerate(target['masks']):
                    mask = mask.cpu().numpy()
                    color = np.random.rand(3)
                    masked_img = np.zeros_like(img_np)
                    for c in range(3):
                        masked_img[:, :, c] = np.where(mask, color[c], 0)
                    axs[i, 2].imshow(masked_img, alpha=0.5)
                    
                    # Draw bounding box
                    box = target['boxes'][j].cpu().numpy()
                    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                        linewidth=2, edgecolor=color, facecolor='none')
                    axs[i, 2].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'))
    plt.close()

def visualize_instances(dataset, indices=None, num_samples=3, save_path=None):
    """
    Visualize extracted instances from ground truth masks to verify correct extraction
    """
    if indices is None:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axs = plt.subplots(len(indices), 2, figsize=(12, 5 * len(indices)))
    if len(indices) == 1:
        axs = np.array([axs])  # Make it 2D for consistent indexing
        
    for i, idx in enumerate(indices):
        image_tensor, target = dataset[idx]
        
        # Convert tensor to numpy for display
        image = image_tensor.permute(1, 2, 0).numpy()
        
        # Display original image
        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis('off')
        
        # Display instances with random colors
        axs[i, 1].imshow(image)
        axs[i, 1].set_title(f"Instances ({len(target['masks'])} found)")
        axs[i, 1].axis('off')
        
        for j, mask in enumerate(target['masks']):
            mask_np = mask.numpy()
            color = np.random.rand(3)
            masked_img = np.zeros_like(image)
            for c in range(3):
                masked_img[:, :, c] = np.where(mask_np, color[c], 0)
            axs[i, 1].imshow(masked_img, alpha=0.5)
            
            # Draw bounding box
            box = target['boxes'][j].numpy()
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                linewidth=2, edgecolor=color, facecolor='none')
            axs[i, 1].add_patch(rect)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Instance visualization saved to {save_path}")
    else:
        plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Enhanced Mask Refinement (from reference.py)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.327062Z","iopub.execute_input":"2025-04-25T08:21:22.327372Z","iopub.status.idle":"2025-04-25T08:21:22.349369Z","shell.execute_reply.started":"2025-04-25T08:21:22.327351Z","shell.execute_reply":"2025-04-25T08:21:22.348498Z"}}
def refine_masks(masks, scores):
    """
    Refines instance masks to prevent overlapping based on the reference.py approach
    
    Args:
        masks: Predicted masks from Mask R-CNN [H, W, N]
        scores: Confidence scores for each mask
        
    Returns:
        refined_masks: Masks with overlap conflicts resolved
    """
    if masks.shape[2] <= 1:  # No need to refine if only one mask
        return masks
        
    # Sort masks by their area (larger objects are likely to be in front)
    areas = np.sum(masks.reshape(-1, masks.shape[2]), axis=0)
    # Sort by score instead if provided
    if scores is not None:
        mask_order = np.argsort(scores)[::-1]  # Higher score first
    else:
        mask_order = np.argsort(areas)[::-1]  # Larger area first
    
    # Initialize union mask to track already covered pixels
    union_mask = np.zeros(masks.shape[:2], dtype=bool)
    refined_masks = masks.copy()
    
    # Process masks in order
    for m in mask_order:
        # Remove pixels that are already covered by higher confidence masks
        refined_masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        # Update the union mask with the current refined mask
        union_mask = np.logical_or(refined_masks[:, :, m], union_mask)
    
    return refined_masks

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Dataset for Instance Segmentation (using pre-computed RLEs)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.350218Z","iopub.execute_input":"2025-04-25T08:21:22.350454Z","iopub.status.idle":"2025-04-25T08:21:22.370511Z","shell.execute_reply.started":"2025-04-25T08:21:22.350423Z","shell.execute_reply":"2025-04-25T08:21:22.369862Z"}}
class RLEFragmentSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, image_list=None, 
                 rle_cache=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.rle_cache = rle_cache or {}  # Use provided RLE cache or empty dict
        
        # Get list of image files
        if image_list is not None:
            self.images = image_list
        else:
            self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_array = np.array(image)
        
        # If testing set, return only the image
        if self.mask_dir is None:
            if self.transform:
                image_array = self.transform(image=image_array)["image"]
            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            return image_tensor, img_name
        
        # For training/validation, prepare masks and boxes using cached RLEs
        if img_name in self.rle_cache:
            cache_data = self.rle_cache[img_name]
            masks = cache_data['instance_masks']
            boxes = cache_data['instance_boxes']
        else:
            # If not in RLE cache, fall back to direct extraction
            # This should rarely happen if preprocessing is done properly
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = np.array(Image.open(mask_path))
            instances = extract_instances_from_mask(mask)
            
            masks = [inst["mask"] for inst in instances]
            boxes = [inst["bbox"] for inst in instances]
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_array, masks=masks)
            image_array = augmented["image"]
            transformed_masks = augmented["masks"]
            
            # Update masks with transformed versions
            masks = transformed_masks
        
        # Prepare for Mask R-CNN format
        labels = [1] * len(masks)  # Class 1 for all fragments
        
        # Convert everything to tensors
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        if not boxes:  # If no instances detected, create a dummy target
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, image_array.shape[0], image_array.shape[1]), dtype=torch.uint8)
            }
        else:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'masks': torch.tensor(np.array(masks), dtype=torch.uint8)
            }
        
        return image_tensor, target

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Data Augmentation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.371341Z","iopub.execute_input":"2025-04-25T08:21:22.371533Z","iopub.status.idle":"2025-04-25T08:21:22.393308Z","shell.execute_reply.started":"2025-04-25T08:21:22.371516Z","shell.execute_reply":"2025-04-25T08:21:22.392531Z"}}
# Define transformations for training
train_transform = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# Define transformations for validation/testing
val_transform = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Split and Create Datasets

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:22.394250Z","iopub.execute_input":"2025-04-25T08:21:22.394509Z","iopub.status.idle":"2025-04-25T08:21:39.015881Z","shell.execute_reply.started":"2025-04-25T08:21:22.394479Z","shell.execute_reply":"2025-04-25T08:21:39.015073Z"}}
# Create datasets using RLE cache instead of instance cache
train_dataset = RLEFragmentSegmentationDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=train_transform,
    image_list=train_images,
    rle_cache=rle_cache if USE_PRECACHED_RLE else None
)

valid_dataset = RLEFragmentSegmentationDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    transform=val_transform,
    image_list=valid_images,
    rle_cache=rle_cache if USE_PRECACHED_RLE else None
)

test_dataset = RLEFragmentSegmentationDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=None,
    transform=val_transform
)

# After creating datasets, visualize instances to verify correct extraction
if not os.path.exists(os.path.join(OUTPUT_DIR, 'instance_extraction.png')):
    print("Visualizing instance extraction results...")
    visualize_instances(
        valid_dataset,
        num_samples=3,
        save_path=os.path.join(OUTPUT_DIR, 'instance_extraction.png')
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Collate Function for Batching Instance Segmentation Data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:39.016803Z","iopub.execute_input":"2025-04-25T08:21:39.017159Z","iopub.status.idle":"2025-04-25T08:21:39.022367Z","shell.execute_reply.started":"2025-04-25T08:21:39.017121Z","shell.execute_reply":"2025-04-25T08:21:39.021563Z"}}
# Custom collate function for instance segmentation
def collate_fn(batch):
    """
    Custom collate function for Mask R-CNN data
    """
    return tuple(zip(*batch))

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mask R-CNN Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:39.023299Z","iopub.execute_input":"2025-04-25T08:21:39.023588Z","iopub.status.idle":"2025-04-25T08:21:44.455239Z","shell.execute_reply.started":"2025-04-25T08:21:39.023552Z","shell.execute_reply":"2025-04-25T08:21:44.454497Z"}}
def get_mask_rcnn_model(num_classes=2):  # Background + Fragment
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the box predictor with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor with a new one for our number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

# Initialize model
model = get_mask_rcnn_model()
model = model.to(DEVICE)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Optimizer and Learning Rate Scheduler

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:44.458033Z","iopub.execute_input":"2025-04-25T08:21:44.458267Z","iopub.status.idle":"2025-04-25T08:21:44.462955Z","shell.execute_reply.started":"2025-04-25T08:21:44.458248Z","shell.execute_reply":"2025-04-25T08:21:44.462136Z"}}
# Set up optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=1,
    eta_min=1e-6
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Training and Validation Functions

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2025-04-25T08:21:44.464249Z","iopub.execute_input":"2025-04-25T08:21:44.464576Z","iopub.status.idle":"2025-04-25T08:21:44.492769Z","shell.execute_reply.started":"2025-04-25T08:21:44.464535Z","shell.execute_reply":"2025-04-25T08:21:44.492147Z"}}
def train_one_epoch(model, optimizer, data_loader, device, scaler=None):
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader):
        # Move images and targets to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        total_loss += losses.item()
    # Return loss and placeholder IoU (actual IoU calculated in validation)
    return total_loss / len(data_loader), 0.0

def validate(model, data_loader, criterion, device):
    """
    Validate the model using IoU score only.
    """
    model.eval()
    iou_scores = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # In eval mode, model always returns predictions even with targets
            # We can't get loss values directly in eval mode
            predictions = model(images)
            
            # Calculate IoU scores
            batch_ious = []
            for i, pred in enumerate(predictions):
                pred_masks = pred['masks']
                
                # Filter based on CONFIDENCE_THRESHOLD
                high_conf_idx = pred['scores'] > CONFIDENCE_THRESHOLD
                filtered_masks = pred_masks[high_conf_idx]
                
                if len(filtered_masks) == 0:
                    # Handle case where no high-confidence masks are predicted
                    binary_gt = torch.zeros((TARGET_SIZE, TARGET_SIZE), device=device, dtype=torch.bool)
                    for mask in targets_gpu[i]['masks']:
                        binary_gt = torch.logical_or(binary_gt, mask > 0)
                    
                    if torch.sum(binary_gt) > 0:  # GT exists but no prediction
                        batch_ious.append(0.0)
                    continue  # Skip if no predictions and no ground truth
                
                # Create binary prediction mask (combining all instances)
                binary_pred = torch.zeros((TARGET_SIZE, TARGET_SIZE), device=device, dtype=torch.bool)
                for mask in filtered_masks:
                    # Handle Mask R-CNN output format [N, 1, H, W]
                    binary_pred = torch.logical_or(binary_pred, mask.squeeze(1) > 0.5)
                
                # Create binary ground truth mask (combining all instances)
                binary_gt = torch.zeros((TARGET_SIZE, TARGET_SIZE), device=device, dtype=torch.bool)
                for mask in targets_gpu[i]['masks']:
                    binary_gt = torch.logical_or(binary_gt, mask > 0)
                
                # Calculate IoU
                img_iou = iou_score(binary_pred.unsqueeze(0).float(), binary_gt.unsqueeze(0).float())
                batch_ious.append(img_iou.item())
            
            if batch_ious:
                iou_scores.append(np.mean(batch_ious))

    # Return a placeholder value for loss (since we can't calculate it in eval mode)
    # and the actual IoU score
    avg_val_iou = np.mean(iou_scores) if iou_scores else 0.0
    return 0.0, avg_val_iou

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs, scaler=None):
    """
    Train model with multi-stage approach (inspired by reference.py)
    
    Args:
        model: The model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        optimizer: The optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: List of epochs for each stage
        scaler: Gradient scaler for mixed precision
    
    Returns:
        history: Dictionary of training history
        best_model_path: Path to the best model
    """
    best_iou = 0.0
    model_save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    
    # For tracking metrics
    train_losses, val_losses, learning_rates = [], [], []
    train_ious, val_ious = [], []
    
    start_time = time.time()
    
    # Stage 1: Train backbone with higher learning rate
    print("Stage 1: Training backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.roi_heads.parameters():
        param.requires_grad = False
    
    # Set higher learning rate for stage 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE * 2
    
    for epoch in range(num_epochs[0]):
        print(f"Epoch {epoch+1}/{num_epochs[0]} (Stage 1)")
        train_loss, _ = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_loss, val_iou = validate(model, valid_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid IoU: {val_iou:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(0.0)  # Use placeholder since we're not calculating training IoU
        val_ious.append(val_iou)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Save best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': val_loss,
                'best_iou': val_iou,
            }, model_save_path)
            print(f"Saved best model to {model_save_path}!")
    
    # Stage 2: Train all layers with normal learning rate
    print("Stage 2: Training all layers...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Reset learning rate for stage 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
    
    for epoch in range(num_epochs[1]):
        print(f"Epoch {epoch+1}/{num_epochs[1]} (Stage 2)")
        train_loss, _ = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_loss, val_iou = validate(model, valid_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid IoU: {val_iou:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(0.0)  # Use placeholder since we're not calculating training IoU
        val_ious.append(val_iou)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + num_epochs[0],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': val_loss,
                'best_iou': val_iou,
            }, model_save_path)
            print(f"Saved best model to {model_save_path}!")
    
    # Stage 3: Fine-tune with lower learning rate
    print("Stage 3: Fine-tuning...")
    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE / 5
    
    for epoch in range(num_epochs[2]):
        print(f"Epoch {epoch+1}/{num_epochs[2]} (Stage 3)")
        train_loss, _ = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_loss, val_iou = validate(model, valid_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f}, Valid IoU: {val_iou:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(0.0)  # Use placeholder since we're not calculating training IoU
        val_ious.append(val_iou)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + num_epochs[0] + num_epochs[1],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': val_loss,
                'best_iou': val_iou,
            }, model_save_path)
            print(f"Saved best model to {model_save_path}!")
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation IoU: {best_iou:.4f}')
    
    return {
        'train_loss': train_losses, 
        'val_loss': val_losses, 
        'train_iou': train_ious, 
        'val_iou': val_ious,
        'lr': learning_rates,
    }, model_save_path

def generate_predictions(model, data_loader, device, is_val=False):
    """
    Generate both instance-level and binary predictions with mask refinement
    """
    model.eval()
    instance_predictions = []
    binary_predictions = []
    metrics = None
    
    if is_val:
        # Initialize evaluation metrics
        instance_ious = []
        binary_ious = []
        
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if is_val:
                images, targets = batch
                # Move targets to CPU for evaluation later
                targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
                image_names = [f"val_{i}" for i in range(len(images))]
            else:
                images, image_names = batch
                
            images = [image.to(device) for image in images]
            
            # Get predictions
            outputs = model(images)
            
            # Process each image
            for i, output in enumerate(outputs):
                img_name = image_names[i]
                if isinstance(img_name, str):
                    image_id = os.path.splitext(img_name)[0]
                else:
                    image_id = f"img_{i}"  # Fallback for validation
                
                # Extract high confidence predictions
                scores = output['scores'].cpu().numpy()
                masks = output['masks'].cpu().numpy()
                
                # Apply threshold to get final masks
                high_conf_idx = scores >= CONFIDENCE_THRESHOLD
                selected_masks = masks[high_conf_idx]
                selected_scores = scores[high_conf_idx]
                
                if len(selected_masks) == 0:
                    # No detections
                    instance_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                    instance_rle = ''
                    binary_rle = ''
                else:
                    # Reshape masks for refinement
                    reshaped_masks = np.zeros((TARGET_SIZE, TARGET_SIZE, len(selected_masks)), dtype=np.uint8)
                    for j, mask in enumerate(selected_masks):
                        # Fix: Handle different mask shapes properly
                        if mask.ndim == 4:  # Shape typically [1, 1, H, W]
                            reshaped_masks[:, :, j] = (mask[0, 0] > 0.5).astype(np.uint8)
                        elif mask.ndim == 3:  # Shape typically [1, H, W]
                            reshaped_masks[:, :, j] = (mask[0] > 0.5).astype(np.uint8)
                        else:  # Other shape - flatten all but last two dimensions
                            flat_mask = mask.reshape(-1, *mask.shape[-2:])
                            reshaped_masks[:, :, j] = (flat_mask[0] > 0.5).astype(np.uint8)
                    
                    # Refine masks to avoid overlaps
                    refined_masks = refine_masks(reshaped_masks, selected_scores)
                    
                    # Instance mask with unique IDs
                    instance_mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                    for j in range(refined_masks.shape[2]):
                        instance_mask = np.where(refined_masks[:, :, j] > 0, j + 1, instance_mask)
                    
                    # Binary mask (all instances are 1)
                    binary_mask = (instance_mask > 0).astype(np.uint8)
                    
                    # Convert to RLE
                    instance_rle = mask_to_rle(instance_mask)
                    binary_rle = mask_to_rle(binary_mask)
                
                # Store predictions
                instance_predictions.append({'id': image_id, 'rle': instance_rle})
                binary_predictions.append({'id': image_id, 'rle': binary_rle})
                
                # Evaluate if validation
                if is_val:
                    target = targets[i]
                    target_masks = target['masks'].numpy()
                    
                    # Create binary ground truth mask
                    target_binary = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                    for mask in target_masks:
                        target_binary = np.logical_or(target_binary, mask).astype(np.uint8)
                    
                    # Calculate IoU for binary prediction
                    if binary_rle != '' and np.sum(target_binary) > 0:
                        binary_mask = (instance_mask > 0).astype(np.uint8)
                        intersection = np.logical_and(binary_mask, target_binary).sum()
                        union = np.logical_or(binary_mask, target_binary).sum()
                        binary_iou = intersection / union if union > 0 else 0
                        binary_ious.append(binary_iou)
                
                # Log number of instances
                if len(selected_masks) > 0:
                    print(f"Image {image_id}: {len(selected_masks)} instances detected")
    
    if is_val and len(binary_ious) > 0:
        metrics = {
            'binary_iou': np.mean(binary_ious),
        }
        print(f"Validation Binary IoU: {metrics['binary_iou']:.4f}")
    
    return instance_predictions, binary_predictions, metrics


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Training Loop - Run the Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-04-25T08:21:44.493709Z","iopub.execute_input":"2025-04-25T08:21:44.493996Z","execution_failed":"2025-04-25T08:21:57.649Z"}}
# Replace the training loop with the multi-stage approach
print("Starting model training...")
history, model_save_path = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    criterion=torch.nn.BCEWithLogitsLoss(),
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    scaler=None
)

# Save training history
save_training_history(
    history['train_loss'],
    history['val_loss'],
    history['lr'],
    history['train_iou'],
    history['val_iou'],
    OUTPUT_DIR
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Visualization and Inference

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-04-25T08:21:57.650Z"}}
# Load best model
print("Loading best model for evaluation...")
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Visualize some predictions
print("Generating visualizations...")
images, targets = next(iter(valid_loader))
visualize_predictions(model, images, targets, DEVICE, num_images=min(5, len(images)))
print(f"Predictions visualization saved to '{os.path.join(OUTPUT_DIR, 'predictions.png')}'")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Generate Instance and Binary Predictions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-04-25T08:21:57.650Z"}}
print("Generating submission files...")
instance_predictions, binary_predictions, _ = generate_predictions(model, test_loader, DEVICE)

# Save instance submission
instance_df = pd.DataFrame(instance_predictions)
instance_submission_path = os.path.join(OUTPUT_DIR, INSTANCE_SUBMISSION_NAME)
instance_df.to_csv(instance_submission_path, index=False)
print(f"Instance segmentation submission saved to '{instance_submission_path}'")

# Save binary submission
binary_df = pd.DataFrame(binary_predictions)
binary_submission_path = os.path.join(OUTPUT_DIR, BINARY_SUBMISSION_NAME)
binary_df.to_csv(binary_submission_path, index=False)
print(f"Binary segmentation submission saved to '{binary_submission_path}'")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Validate with Both Metrics

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"execution_failed":"2025-04-25T08:21:57.650Z"}}
print("Evaluating on validation set...")
_, _, val_metrics = generate_predictions(model, valid_loader, DEVICE, is_val=True)
print(f"Validation metrics: {val_metrics}")

# Visualize some predictions with instance coloring
def visualize_instance_predictions(model, images, targets=None, device=DEVICE, num_images=3):
    """Visualize predictions with each instance colored differently"""
    model.eval()
    fig, axs = plt.subplots(num_images, 3 if targets else 2, figsize=(15, 5 * num_images))
    
    with torch.no_grad():
        for i, image in enumerate(images[:num_images]):
            # Forward pass
            image_tensor = image.to(device)
            output = model([image_tensor])[0]
            
            # Plot original image
            img_np = image.permute(1, 2, 0).cpu().numpy()
            axs[i, 0].imshow(img_np)
            axs[i, 0].set_title("Original Image")
            axs[i, 0].axis('off')
            
            # Plot instance predictions
            axs[i, 1].imshow(img_np)
            axs[i, 1].set_title("Instance Predictions")
            axs[i, 1].axis('off')
            
            # Create instance mask where each instance has a different random color
            instance_viz = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
            for j, mask in enumerate(output['masks']):
                if output['scores'][j] > VISUALIZATION_THRESHOLD:  # Only show high confidence predictions
                    mask = mask.squeeze(1).cpu().numpy() > 0.5  # Fixed: using squeeze(1)
                    color = np.random.rand(3)
                    for c in range(3):
                        instance_viz[:, :, c] += np.where(mask, color[c] * 0.7, 0)
            
            # Overlay on image
            overlay = img_np * 0.6 + instance_viz * 0.4
            overlay = np.clip(overlay, 0, 1)
            axs[i, 1].imshow(overlay)
            
            # Show ground truth if available
            if targets:
                axs[i, 2].imshow(img_np)
                axs[i, 2].set_title("Ground Truth")
                axs[i, 2].axis('off')
                
                # Create ground truth instance visualization
                gt_viz = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
                for j, mask in enumerate(targets[i]['masks']):
                    mask_np = mask.cpu().numpy()
                    color = np.random.rand(3)
                    for c in range(3):
                        gt_viz[:, :, c] += np.where(mask_np, color[c] * 0.7, 0)
                
                # Overlay on image
                gt_overlay = img_np * 0.6 + gt_viz * 0.4
                gt_overlay = np.clip(gt_overlay, 0, 1)
                axs[i, 2].imshow(gt_overlay)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'instance_predictions.png'))
    plt.close()
    print(f"Instance visualization saved to '{os.path.join(OUTPUT_DIR, 'instance_predictions.png')}'")

# Visualize instances
print("Generating detailed instance visualizations...")
visualize_instance_predictions(model, images, targets, DEVICE, num_images=min(5, len(images)))

print("All processing completed successfully!")