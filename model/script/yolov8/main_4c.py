import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import shutil
import cv2
from copy import deepcopy
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Config:
    def __init__(self):
        # Model settings
        self.model_name = 'yolov8s-seg.pt'  # Pre-trained YOLOv8s segmentation model
        self.modified_model_path = '/kaggle/working/yolov8s-seg-4c.pt'  # Path to save modified model
        
        # Paths
        self.data_yaml = '/kaggle/working/dataset.yaml'  # Path to dataset config
        self.output_path = '/kaggle/working/'
        self.project = '/kaggle/working/runs/segment'
        self.name = 'fragment_seg_4c'
        
        # Training parameters
        self.epochs = 10
        self.imgsz = 512  # Specified image size: 512x512
        self.batch_size = 16
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # 4-Channel Configuration
        self.in_channels = 4  # Number of input channels (RGB + depth)
        self.depth_channel_weight = 1.0  # Weight for initializing the depth channel
        
        # Class information
        self.classes = ['fragment']  # Single class: fragment

def modify_model_for_4c(model, config):
    """
    Modify the model's first convolutional layer to accept 4-channel input.
    This preserves pre-trained weights for RGB channels and initializes the 4th channel.
    """
    print(f"Modifying model to accept {config.in_channels}-channel input...")
    
    # Get the first layer
    conv1 = model.model.model[0][0]
    
    # Check current input channels
    current_in_channels = conv1.in_channels
    if current_in_channels == config.in_channels:
        print(f"Model already accepts {config.in_channels} channels. No modification needed.")
        return model
    
    # Get attributes of the first conv layer
    out_channels = conv1.out_channels
    kernel_size = conv1.kernel_size
    stride = conv1.stride
    padding = conv1.padding
    bias = conv1.bias is not None
    
    # Create a new conv layer with 4 channels input
    new_conv = torch.nn.Conv2d(
        config.in_channels, 
        out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias
    )
    
    # Load existing weights for RGB channels
    with torch.no_grad():
        # Copy the weights for the RGB channels
        new_conv.weight[:, :3, :, :] = conv1.weight.clone()
        
        # Initialize the 4th channel 
        # Option 1: Initialize with zeros
        # new_conv.weight[:, 3:, :, :] = 0
        
        # Option 2: Initialize with mean of RGB channels
        new_conv.weight[:, 3:, :, :] = conv1.weight.mean(dim=1, keepdim=True) * config.depth_channel_weight
        
        # Copy bias if it exists
        if bias:
            new_conv.bias = torch.nn.Parameter(conv1.bias.clone())
    
    # Replace the first layer
    model.model.model[0][0] = new_conv
    
    print(f"Successfully modified first layer to accept {config.in_channels} input channels")
    print(f"New first layer shape: {model.model.model[0][0].weight.shape}")
    
    return model

def visualize_sample_4c(image_path, depth_path=None, model=None):
    """Visualize a 4-channel sample with RGB + depth"""
    plt.figure(figsize=(15, 10))
    
    # Load and display RGB image
    plt.subplot(2, 2, 1)
    rgb_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis('off')
    
    # Load and display depth image if provided
    if depth_path and os.path.exists(depth_path):
        plt.subplot(2, 2, 2)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(depth_img, cmap='jet')
        plt.title("Depth Channel")
        plt.axis('off')
    
    # If model is provided, predict and show results
    if model and os.path.exists(image_path):
        # For visualization, we'll just use the RGB image
        plt.subplot(2, 2, 3)
        results = model.predict(image_path)
        result = results[0]
        plt.imshow(result.plot())
        plt.title("Prediction")
        plt.axis('off')
        
        # If we have a 4C image for actual prediction
        if depth_path and os.path.exists(depth_path):
            plt.subplot(2, 2, 4)
            # This is where you'd process and combine the 4-channel input
            # For demonstration purposes, we're just showing expected outputs
            plt.text(0.5, 0.5, "4-Channel Prediction\n(Combined RGB + Depth)", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, transform=plt.gca().transAxes)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def preprocess_4c_batch(rgb_paths, depth_paths, imgsz=512):
    """
    Preprocess a batch of 4-channel images (RGB + depth).
    Returns a tensor ready for model inference.
    """
    batch = []
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        # Read RGB image
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (imgsz, imgsz))
        
        # Read depth image
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (imgsz, imgsz))
        
        # Normalize
        rgb_img = rgb_img.astype(np.float32) / 255.0
        depth_img = depth_img.astype(np.float32) / 255.0
        
        # Combine channels
        combined = np.zeros((imgsz, imgsz, 4), dtype=np.float32)
        combined[:, :, :3] = rgb_img
        combined[:, :, 3] = depth_img
        
        # HWC to CHW format
        combined = combined.transpose(2, 0, 1)
        batch.append(combined)
    
    # Stack into batch
    batch = np.stack(batch, axis=0)
    return torch.from_numpy(batch)

def find_depth_path(rgb_path):
    """Find corresponding depth image for an RGB image"""
    # Method 1: Same filename with _depth suffix
    dir_path = os.path.dirname(rgb_path)
    filename = Path(rgb_path).stem
    depth_path = os.path.join(dir_path, f"{filename}_depth.png")
    
    if os.path.exists(depth_path):
        return depth_path
    
    # Method 2: Same filename in a depth subdirectory
    depth_dir = os.path.join(os.path.dirname(dir_path), "depth")
    if os.path.exists(depth_dir):
        depth_path = os.path.join(depth_dir, os.path.basename(rgb_path))
        if os.path.exists(depth_path):
            return depth_path
    
    # Method 3: Replace 'images' with 'depth' in path
    depth_path = rgb_path.replace("images", "depth")
    if os.path.exists(depth_path):
        return depth_path
    
    return None

def train_model_4c(config):
    """Train YOLOv8 segmentation model with 4-channel input."""
    print(f"Training YOLOv8 segmentation model for class: {config.classes[0]} with {config.in_channels}-channel input")
    print(f"Data path: {config.data_yaml}")
    print(f"Output path: {config.output_path}")
    print(f"Image size: {config.imgsz}x{config.imgsz}")
    
    # Load pretrained model
    model = YOLO(config.model_name)
    
    # Modify model for 4-channel input
    model = modify_model_for_4c(model, config)
    
    # Save modified model before training
    model.export(format="pt", imgsz=config.imgsz)
    if os.path.exists(model.ckpt_path):
        shutil.copy(model.ckpt_path, config.modified_model_path)
        print(f"Saved modified model to {config.modified_model_path}")
    
    # Train model
    print(f"Starting training for {config.epochs} epochs with image size {config.imgsz}...")
    results = model.train(
        data=config.data_yaml,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch_size,
        device=config.device,
        project=config.project,
        name=config.name,
        seed=config.seed
    )
    
    # Plot training results
    metrics = results.results_dict
    plt.figure(figsize=(15, 5))
    
    # Plot mAP
    plt.subplot(1, 3, 1)
    plt.plot(metrics.get('metrics/mAP50(B)', []), label='mAP50')
    plt.plot(metrics.get('metrics/mAP50-95(B)', []), label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Metrics')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(metrics.get('train/box_loss', []), label='Box Loss')
    plt.plot(metrics.get('train/seg_loss', []), label='Seg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 3, 3)
    plt.plot(metrics.get('val/box_loss', []), label='Val Box Loss')
    plt.plot(metrics.get('val/seg_loss', []), label='Val Seg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, "training_metrics_4c.png"))
    plt.show()
    
    return model, results

def evaluate_model_4c(model, config):
    """Evaluate the trained 4-channel model on validation data."""
    print("Running validation...")
    val_results = model.val()
    print("Validation Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # Save model
    best_model_path = os.path.join(config.project, config.name, "weights/best.pt")
    save_path = os.path.join(config.output_path, "best_4c.pt")
    shutil.copy(best_model_path, save_path)
    print(f"Best model saved to: {save_path}")
    
    # Visualize a sample prediction
    print("Visualizing sample prediction...")
    dataset_dir = os.path.dirname(config.data_yaml)
    val_img_dir = os.path.join(os.path.dirname(config.data_yaml), 'val/images')
    
    if os.path.exists(val_img_dir):
        sample_images = os.listdir(val_img_dir)
        if sample_images:
            sample_image = os.path.join(val_img_dir, sample_images[0])
            depth_image = find_depth_path(sample_image)
            visualize_sample_4c(sample_image, depth_image, model)
        else:
            print("No validation images found for visualization")
    else:
        print(f"Validation image directory not found at {val_img_dir}")
    
    return val_results

def custom_predict_4c(model, rgb_path, depth_path, imgsz=512):
    """Custom prediction function for 4-channel input"""
    # Read and preprocess RGB image
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    orig_shape = rgb_img.shape[:2]  # Store original shape for later
    rgb_img = cv2.resize(rgb_img, (imgsz, imgsz))
    
    # Read and preprocess depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth_img = cv2.resize(depth_img, (imgsz, imgsz))
    
    # Normalize
    rgb_norm = rgb_img.astype(np.float32) / 255.0
    depth_norm = depth_img.astype(np.float32) / 255.0
    
    # Combine channels
    combined = np.zeros((imgsz, imgsz, 4), dtype=np.float32)
    combined[:, :, :3] = rgb_norm
    combined[:, :, 3] = depth_norm
    
    # HWC to CHW format
    combined = combined.transpose(2, 0, 1)
    
    # Create batch dimension
    batch = np.expand_dims(combined, axis=0)
    
    # Convert to torch tensor
    batch_tensor = torch.from_numpy(batch).to(model.device)
    
    # Perform inference
    with torch.no_grad():
        results = model.predict(batch_tensor, imgsz=imgsz)
    
    return results[0]

def main():
    """Main function."""
    # Initialize configuration
    config = Config()
    
    # Train model with 4-channel input
    model, results = train_model_4c(config)
    
    # Evaluate model
    evaluate_model_4c(model, config)
    
    print("Training and evaluation of 4-channel model completed!")

if __name__ == '__main__':
    main()
