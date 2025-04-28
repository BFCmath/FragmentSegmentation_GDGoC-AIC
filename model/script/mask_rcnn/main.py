# Imports
import os
import time
import random
import collections
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Configuration class with all settings
class Config:
    # Paths for Kaggle environment
    TRAIN_IMAGES_DIR = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images"
    TRAIN_MASKS_DIR = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/masks"
    TEST_IMAGES_DIR = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/test/images"
    CACHE_DIR = "/kaggle/working/cache"
    MODEL_SAVE_DIR = "/kaggle/working"
    
    # Image dimensions
    WIDTH = 512
    HEIGHT = 512
    
    # Dataset split
    VALIDATION_RATIO = 0.15
    RANDOM_SEED = 42
    
    # Training parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 8
    LEARNING_RATE = 0.0005  # Reduced for Adam optimizer
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Model parameters
    MASK_THRESHOLD = 0.5
    BOX_DETECTIONS_PER_IMG = 250
    MIN_SCORE = 0.59
    
    # Other settings
    NORMALIZE = False
    USE_SCHEDULER = True
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Warmup settings
    WARMUP_EPOCHS = 1
    WARMUP_FACTOR = 0.001
    
    # ImageNet mean and std for normalization
    RESNET_MEAN = (0.485, 0.456, 0.406)
    RESNET_STD = (0.229, 0.224, 0.225)

# Fix randomness for reproducibility
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)

# Transformation classes
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            
            # Fix any boxes with zero height after flip
            zero_height_indices = bbox[:, 1] == bbox[:, 3]
            if torch.any(zero_height_indices):
                bbox[zero_height_indices, 3] += 1
                
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            
            # Fix any boxes with zero width after flip
            zero_width_indices = bbox[:, 0] == bbox[:, 2]
            if torch.any(zero_width_indices):
                bbox[zero_width_indices, 2] += 1
                
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, Config.RESNET_MEAN, Config.RESNET_STD)
        return image, target

# Add a box validation function
def validate_boxes(boxes):
    """Ensure all boxes have positive width and height"""
    if isinstance(boxes, torch.Tensor):
        # For PyTorch tensors
        zero_width = boxes[:, 0] == boxes[:, 2]
        zero_height = boxes[:, 1] == boxes[:, 3]
        
        if torch.any(zero_width):
            boxes[zero_width, 2] += 1
        if torch.any(zero_height):
            boxes[zero_height, 3] += 1
    else:
        # For numpy arrays
        boxes = np.array(boxes)
        zero_width = boxes[:, 0] == boxes[:, 2]
        zero_height = boxes[:, 1] == boxes[:, 3]
        
        if np.any(zero_width):
            boxes[zero_width, 2] += 1
        if np.any(zero_height):
            boxes[zero_height, 3] += 1
            
    return boxes

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            # Validate boxes after conversion to tensor
            target["boxes"] = validate_boxes(target["boxes"])
        return image, target
    
def get_transform(train):
    transforms = [ToTensor()]
    if Config.NORMALIZE:
        transforms.append(Normalize())
    
    # Data augmentation for train
    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)

# Utility functions
def mask_to_rle(mask):
    """
    Convert a binary mask to run-length encoding (RLE)
    """
    # Flatten mask
    pixels = mask.flatten()
    # Compress the mask with RLE
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    # Convert to string
    rle = ' '.join(str(x) for x in runs)
    return rle

def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask

# Preprocessing utility to extract and cache masks
def preprocess_and_cache_masks(image_dir, mask_dir, cache_dir):
    """Extract mask information and save to disk for faster loading"""
    os.makedirs(cache_dir, exist_ok=True)
    
    image_ids = [f[:-4] for f in os.listdir(image_dir)]
    
    for image_id in image_ids:
        cache_file = os.path.join(cache_dir, f"{image_id}.pkl")
        
        # Skip if already cached
        if os.path.exists(cache_file):
            continue
            
        mask_path = os.path.join(mask_dir, image_id + '.png')
        colored_mask = np.array(Image.open(mask_path))
        
        # Extract individual masks
        if len(colored_mask.shape) == 3:  # RGB mask
            h, w, c = colored_mask.shape
            reshaped_mask = colored_mask.reshape(-1, c)
            unique_colors = np.unique(reshaped_mask, axis=0)
            # Remove background (assuming it's black [0,0,0])
            unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]
            
            masks = []
            boxes = []
            for color in unique_colors:
                # Create binary mask for this color
                mask = np.all(colored_mask == color.reshape(1, 1, 3), axis=2).astype(np.uint8)
                masks.append(mask)
                # Get bounding box
                pos = np.where(mask)
                if len(pos[0]) > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    
                    # Ensure boxes have at least 1 pixel height and width
                    if xmin == xmax:
                        xmax = xmin + 1
                    if ymin == ymax:
                        ymax = ymin + 1
                        
                    boxes.append([xmin, ymin, xmax, ymax])
        else:  # Grayscale mask
            h, w = colored_mask.shape
            unique_values = np.unique(colored_mask)
            # Remove background (assuming it's 0)
            unique_values = unique_values[unique_values > 0]
            
            masks = []
            boxes = []
            for value in unique_values:
                mask = (colored_mask == value).astype(np.uint8)
                masks.append(mask)
                # Get bounding box
                pos = np.where(mask)
                if len(pos[0]) > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    
                    # Ensure boxes have at least 1 pixel height and width
                    if xmin == xmax:
                        xmax = xmin + 1
                    if ymin == ymax:
                        ymax = ymin + 1
                        
                    boxes.append([xmin, ymin, xmax, ymax])
        
        # All fragments are class 1
        labels = np.ones(len(masks), dtype=np.int64)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'masks': np.array(masks) if masks else np.zeros((0, h, w), dtype=np.uint8),
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                'labels': labels
            }, f)
    
    print(f"Preprocessing complete. Cached {len(image_ids)} masks.")

# Dataset for training that loads from cache
class FragmentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, cache_dir=None, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cache_dir = cache_dir
        self.image_ids = [f[:-4] for f in os.listdir(image_dir)]
        
        # Create cache directory if needed
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Preprocess and cache all masks
            preprocess_and_cache_masks(image_dir, mask_dir, cache_dir)
        
    def __getitem__(self, idx):
        # Load image and mask
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id + '.jpg')
        img = Image.open(img_path).convert("RGB")
        
        # Load cached mask data if available
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f"{image_id}.pkl")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            masks = cache_data['masks']
            boxes = cache_data['boxes']
            labels = cache_data['labels']
        else:
            # Original extraction code as fallback
            mask_path = os.path.join(self.mask_dir, image_id + '.png')
            colored_mask = np.array(Image.open(mask_path))
            
            # Extract individual masks from colored mask
            h, w, c = colored_mask.shape
            reshaped_mask = colored_mask.reshape(-1, c)
            unique_colors = np.unique(reshaped_mask, axis=0)
            unique_colors = unique_colors[~np.all(unique_colors == 0, axis=1)]
            
            masks = []
            boxes = []
            for color in unique_colors:
                mask = np.all(colored_mask == color.reshape(1, 1, 3), axis=2).astype(np.uint8)
                masks.append(mask)
                pos = np.where(mask)
                if len(pos[0]) > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    
                    # Ensure boxes have at least 1 pixel height and width
                    if xmin == xmax:
                        xmax = xmin + 1
                    if ymin == ymax:
                        ymax = ymin + 1
                        
                    boxes.append([xmin, ymin, xmax, ymax])
            
            labels = np.ones(len(masks), dtype=np.int64)
        
        # Convert to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)
        
        # Target dictionary for Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        # Final validation of boxes before transforms
        target['boxes'] = validate_boxes(target['boxes'])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        # Final check after transforms
        target['boxes'] = validate_boxes(target['boxes'])
        
        return img, target
    
    def __len__(self):
        return len(self.image_ids)

# Dataset for testing
class FragmentTestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [f[:-4] for f in os.listdir(image_dir)]
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id + '.jpg')
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms is not None:
            img, _ = self.transforms(img, None)
            
        return {'image': img, 'image_id': image_id}
    
    def __len__(self):
        return len(self.image_ids)

# Get Mask R-CNN model
def get_model():
    # 2 classes: background (0) and fragment (1)
    NUM_CLASSES = 2
    
    if Config.NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, 
            box_detections_per_img=Config.BOX_DETECTIONS_PER_IMG,
            image_mean=Config.RESNET_MEAN, 
            image_std=Config.RESNET_STD
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
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

# Custom learning rate scheduler with warmup
class WarmupLR:
    def __init__(self, optimizer, warmup_epochs, warmup_factor, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            scale_factor = self.warmup_factor * (1 - alpha) + alpha
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * scale_factor
        elif self.after_scheduler is not None:
            self.after_scheduler.step()
            
        self.current_epoch += 1
        
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# Function to evaluate model on validation set
def evaluate_model(model, data_loader, device):
    # Save original model state
    was_training = model.training
    
    # Set to training mode to get losses
    model.train()
    
    loss_accum = 0.0
    loss_mask_accum = 0.0
    n_batches = len(data_loader)
    
    if n_batches == 0:
        return 0, 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask
    
    # Restore original training mode
    if not was_training:
        model.eval()
        
    val_loss = loss_accum / n_batches
    val_loss_mask = loss_mask_accum / n_batches
    
    return val_loss, val_loss_mask

# Function to calculate IoU for evaluation
def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU between predicted and ground truth mask"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0
    return intersection / union

# Evaluate model performance on validation set
def evaluate_val_performance(model, dataset, device):
    model.eval()
    ious = []
    
    print("Evaluating performance on validation set...")
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]
        
        # Process predicted masks
        pred_combined_mask = np.zeros((img.shape[1], img.shape[2]))
        previous_masks = []
        
        for i, mask in enumerate(prediction['masks']):
            # Filter low-confidence predictions
            score = prediction["scores"][i].cpu().item()
            if score < Config.MIN_SCORE:
                continue
                
            mask = mask.cpu().numpy()
            binary_mask = mask[0] > Config.MASK_THRESHOLD
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)
            pred_combined_mask = np.logical_or(pred_combined_mask, binary_mask)
        
        # Get ground truth mask (combine all instance masks)
        gt_combined_mask = np.zeros((img.shape[1], img.shape[2]))
        for mask in target['masks']:
            gt_combined_mask = np.logical_or(gt_combined_mask, mask.numpy())
        
        # Calculate IoU
        iou = calculate_iou(pred_combined_mask, gt_combined_mask)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    print(f"Validation IoU: {mean_iou:.4f}")
    return mean_iou

# Analyze prediction results
def analyze_sample(model, dataset, sample_index):
    img, targets = dataset[sample_index]
    
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(1, 3, 1)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title("Image")
    plt.axis('off')
    
    # Show ground truth masks
    plt.subplot(1, 3, 2)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    all_masks = np.zeros((img.shape[1], img.shape[2]))
    for mask in targets['masks']:
        all_masks = np.logical_or(all_masks, mask.numpy())
    plt.imshow(all_masks, alpha=0.4)
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Show predictions
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(Config.DEVICE)])[0]
    
    plt.subplot(1, 3, 3)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    all_pred_masks = np.zeros((img.shape[1], img.shape[2]))
    previous_masks = []
    
    for i, mask in enumerate(prediction['masks']):
        score = prediction["scores"][i].cpu().item()
        if score < Config.MIN_SCORE:
            continue
        mask = mask.cpu().numpy()
        binary_mask = mask[0] > Config.MASK_THRESHOLD
        binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
        previous_masks.append(binary_mask)
        all_pred_masks = np.logical_or(all_pred_masks, binary_mask)
    
    plt.imshow(all_pred_masks, alpha=0.4)
    plt.title("Predictions")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualization function
def visualize_predictions(model, val_dataset, test_dataset, device, num_samples=5, output_dir=None):
    """
    Visualize predictions on validation and test images
    
    Parameters:
    model (torch.nn.Module): The model to use for predictions
    val_dataset (torch.utils.data.Subset): Validation dataset subset
    test_dataset (FragmentTestDataset): Test dataset
    device (torch.device): Device to run the model on
    num_samples (int): Number of samples to visualize
    output_dir (str): Directory to save visualizations
    """
    model.eval()
    
    # Create figure for validation images
    fig1, axes1 = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    fig1.suptitle('Validation Images: Predictions vs Ground Truth', fontsize=16)
    
    # Create figure for test images
    fig2, axes2 = plt.subplots(num_samples, 1, figsize=(6, 3*num_samples))
    fig2.suptitle('Test Images: Predictions', fontsize=16)
    
    # Get random validation sample indices
    val_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    # Visualize validation samples
    for i, idx in enumerate(val_indices):
        img, target = val_dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]
        
        # Process image for display
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Process ground truth masks - create a colorful mask
        gt_combined_mask = np.zeros((img.shape[1], img.shape[2], 3))
        for j, mask in enumerate(target['masks']):
            # Create a random color for this mask
            color = np.random.rand(3) * 0.8 + 0.2  # Brighter colors
            mask_np = mask.cpu().numpy()
            # Add this mask to the combined mask with its color
            for c in range(3):
                gt_combined_mask[:, :, c] = np.where(mask_np == 1, color[c], gt_combined_mask[:, :, c])
        
        # Process predicted masks - create a colorful mask
        pred_combined_mask = np.zeros((img.shape[1], img.shape[2], 3))
        previous_masks = []
        
        for i_mask, mask in enumerate(prediction['masks']):
            # Filter low-confidence predictions
            score = prediction["scores"][i_mask].cpu().item()
            if score < Config.MIN_SCORE:
                continue
            
            # Create a random color for this mask
            color = np.random.rand(3) * 0.8 + 0.2  # Brighter colors
            mask_np = mask.cpu().numpy()
            binary_mask = mask_np[0] > Config.MASK_THRESHOLD
            binary_mask = remove_overlapping_pixels(binary_mask.copy(), previous_masks)
            previous_masks.append(binary_mask)
            
            # Add this mask to the combined mask with its color
            for c in range(3):
                pred_combined_mask[:, :, c] = np.where(binary_mask, color[c], pred_combined_mask[:, :, c])
        
        # Display images and masks
        axes1[i, 0].imshow(img_np)
        axes1[i, 0].imshow(pred_combined_mask, alpha=0.5)
        axes1[i, 0].set_title("Prediction")
        axes1[i, 0].axis('off')
        
        axes1[i, 1].imshow(img_np)
        axes1[i, 1].imshow(gt_combined_mask, alpha=0.5)
        axes1[i, 1].set_title("Ground Truth")
        axes1[i, 1].axis('off')
    
    # Get random test samples
    test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    # Visualize test samples
    for i, idx in enumerate(test_indices):
        sample = test_dataset[idx]
        img = sample['image']
        image_id = sample['image_id']
        
        # Get prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]
        
        # Process image for display
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Process predicted masks - create a colorful mask
        pred_combined_mask = np.zeros((img.shape[1], img.shape[2], 3))
        previous_masks = []
        
        for i_mask, mask in enumerate(prediction['masks']):
            # Filter low-confidence predictions
            score = prediction["scores"][i_mask].cpu().item()
            if score < Config.MIN_SCORE:
                continue
            
            # Create a random color for this mask
            color = np.random.rand(3) * 0.8 + 0.2  # Brighter colors
            mask_np = mask.cpu().numpy()
            binary_mask = mask_np[0] > Config.MASK_THRESHOLD
            binary_mask = remove_overlapping_pixels(binary_mask.copy(), previous_masks)
            previous_masks.append(binary_mask)
            
            # Add this mask to the combined mask with its color
            for c in range(3):
                pred_combined_mask[:, :, c] = np.where(binary_mask, color[c], pred_combined_mask[:, :, c])
        
        # Display image and mask
        axes2[i].imshow(img_np)
        axes2[i].imshow(pred_combined_mask, alpha=0.5)
        axes2[i].set_title(f"Test Image: {image_id}")
        axes2[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for titles
    
    # Save figures
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'validation_predictions.png'))
        fig2.savefig(os.path.join(output_dir, 'test_predictions.png'))
        print(f"Saved visualizations to {output_dir}")
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create dataset with caching
    full_dataset = FragmentDataset(
        Config.TRAIN_IMAGES_DIR, 
        Config.TRAIN_MASKS_DIR,
        cache_dir=Config.CACHE_DIR
    )
    
    # Create train/validation split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(Config.VALIDATION_RATIO * dataset_size))
    
    # Shuffle indices
    fix_all_seeds(Config.RANDOM_SEED)  # Use fixed seed for reproducible splits
    np.random.shuffle(indices)
    
    # Split indices
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create data subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transforms = get_transform(train=True)
    val_dataset.dataset.transforms = get_transform(train=False)
    
    print(f"Training on {len(train_indices)} samples, validating on {len(val_indices)} samples")
    
    # Create data loaders
    dl_train = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    dl_val = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    model = get_model()
    model.to(Config.DEVICE)
    
    # Enable gradient for all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    model.train()
    
    # Use Adam optimizer instead of SGD
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Store initial learning rate for warmup scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    
    # Learning rate scheduler with warmup
    base_lr_scheduler = None
    if Config.USE_SCHEDULER:
        base_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
    lr_scheduler = WarmupLR(
        optimizer, 
        warmup_epochs=Config.WARMUP_EPOCHS, 
        warmup_factor=Config.WARMUP_FACTOR,
        after_scheduler=base_lr_scheduler if Config.USE_SCHEDULER else None
    )
        
    # Training loop
    n_batches = len(dl_train)
    best_val_loss = float('inf')  # Initialize with infinity to ensure first model is saved

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch} of {Config.NUM_EPOCHS}")
        print(f"Learning rate: {lr_scheduler.get_lr()[0]:.6f}")
        
        # Training phase
        model.train()
        time_start = time.time()
        loss_accum = 0.0
        loss_mask_accum = 0.0
        
        for batch_idx, (images, targets) in enumerate(dl_train, 1):
            # Predict
            images = list(image.to(Config.DEVICE) for image in images)
            targets = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask
            
            if batch_idx % 50 == 0:
                print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] "
                      f"Batch train loss: {loss.item():7.3f}. "
                      f"Mask-only loss: {loss_mask:7.3f}")
        
        # Step scheduler after each epoch
        lr_scheduler.step()
        
        # Train losses
        train_loss = loss_accum / n_batches
        train_loss_mask = loss_mask_accum / n_batches
        
        # Validation phase
        val_loss, val_loss_mask = evaluate_model(model, dl_val, Config.DEVICE)
        
        elapsed = time.time() - time_start
        
        # Save model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(Config.MODEL_SAVE_DIR, "fragment_model_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"    New best model saved! Validation loss: {best_val_loss:.4f}")
        
        # Print epoch summary
        prefix = f"[Epoch {epoch:2d} / {Config.NUM_EPOCHS:2d}]"
        print(f"{prefix} Train mask loss: {train_loss_mask:7.3f}, Val mask loss: {val_loss_mask:7.3f}")
        print(f"{prefix} Train loss: {train_loss:7.3f}, Val loss: {val_loss:7.3f}, [{elapsed:.0f} secs]")
    
    # Load best model for prediction
    best_model_path = os.path.join(Config.MODEL_SAVE_DIR, "fragment_model_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for prediction")
    
    # Evaluate on validation set first
    print("Evaluating best model on validation set...")
    val_dataset.dataset.transforms = get_transform(train=False)
    final_val_iou = evaluate_val_performance(model, val_dataset, Config.DEVICE)
    print(f"Best model validation IoU: {final_val_iou:.4f}")
    
    # Visualize predictions on validation and test samples
    print("Generating visualizations...")
    ds_test = FragmentTestDataset(Config.TEST_IMAGES_DIR, transforms=get_transform(train=False))
    visualization_dir = os.path.join(Config.MODEL_SAVE_DIR, "visualizations")
    visualize_predictions(model, val_dataset, ds_test, Config.DEVICE, num_samples=5, output_dir=visualization_dir)
    
    # Make predictions on test set
    model.eval()
    print("Generating test predictions...")
    submission = []
    for sample in ds_test:
        img = sample['image']
        image_id = sample['image_id']
        
        with torch.no_grad():
            result = model([img.to(Config.DEVICE)])[0]
        
        # Create a single combined mask for the entire image
        combined_mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
        previous_masks = []
        
        for i, mask in enumerate(result["masks"]):
            # Filter-out low-scoring results
            score = result["scores"][i].cpu().item()
            if score < Config.MIN_SCORE:
                continue
            
            mask = mask.cpu().numpy()
            # Keep only highly likely pixels
            binary_mask = mask[0] > Config.MASK_THRESHOLD
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)
            
            # Combine with the full image mask
            combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
        
        # Generate a single RLE for the entire image's combined mask
        rle = mask_to_rle(combined_mask)
        
        # Add a single entry for this image
        submission.append((image_id, rle))

    # Save submission file
    submission_path = os.path.join(Config.MODEL_SAVE_DIR, "submission.csv")
    df_sub = pd.DataFrame(submission, columns=['id', 'rle'])  # Use 'rle' instead of 'predicted'
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission created at {submission_path} with {len(df_sub)} predictions")
    print(f"Final submission contains {len(df_sub['id'].unique())} unique image IDs")
