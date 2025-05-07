import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import shutil

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Config:
    def __init__(self):
        # Model settings
        self.model_name = 'yolov8s-seg.pt'  # Pre-trained YOLOv8n segmentation model
        
        # Paths
        self.data_yaml = '/kaggle/working/dataset.yaml'  # Path to dataset config
        self.output_path = '/kaggle/working/'
        self.project = '/kaggle/working/runs/segment'
        self.name = 'fragment_seg'
        
        # Training parameters
        self.epochs = 10
        self.imgsz = 512  # Specified image size: 512x512
        self.batch_size = 16
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # Class information
        self.classes = ['fragment']  # Single class: fragment

def visualize_sample(image_path, model=None):
    """Visualize a sample image with predictions if a model is provided"""
    plt.figure(figsize=(10, 10))
    
    # Display original image
    plt.subplot(1, 2 if model else 1, 1)
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # If model is provided, show prediction
    if model:
        results = model.predict(image_path)
        result = results[0]
        
        plt.subplot(1, 2, 2)
        plt.imshow(result.plot())
        plt.title("Prediction")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_model(config):
    """Train YOLOv8 segmentation model."""
    print(f"Training YOLOv8 segmentation model for class: {config.classes[0]}")
    print(f"Data path: {config.data_yaml}")
    print(f"Output path: {config.output_path}")
    print(f"Image size: {config.imgsz}x{config.imgsz}")
    
    # Load model
    model = YOLO(config.model_name)
    
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
    plt.savefig(os.path.join(config.output_path, "training_metrics.png"))
    plt.show()
    
    return model, results

def evaluate_model(model, config):
    """Evaluate the trained model on validation data."""
    print("Running validation...")
    val_results = model.val()
    print("Validation Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # Save model
    best_model_path = os.path.join(config.project, config.name, "weights/best.pt")
    save_path = os.path.join(config.output_path, "best.pt")
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
            visualize_sample(sample_image, model)
        else:
            print("No validation images found for visualization")
    else:
        print(f"Validation image directory not found at {val_img_dir}")
    
    return val_results

def main():
    """Main function."""
    # Initialize configuration
    config = Config()
    
    # Train model
    model, results = train_model(config)
    
    # Evaluate model
    evaluate_model(model, config)
    
    print("Training and evaluation completed!")

if __name__ == '__main__':
    main()