import os
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from tqdm import tqdm

class Config:
    """Configuration parameters for dataset preparation."""
    # Data split configuration
    VAL_SPLIT = 0.2  # Percentage of data for validation

    # Source paths
    RGB_SOURCE_PATH = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images"
    DEPTH_SOURCE_PATH = "/kaggle/input/fst-depth-map/vits"
    LABEL_SOURCE_PATH = "/kaggle/input/fst-mask-convert"
    
    # Target paths
    OUTPUT_BASE_PATH = "/kaggle/working/data"
    
    # File patterns
    RGB_PATTERN = "*.jpg"
    DEPTH_PATTERN = "*_depth.png"
    LABEL_PATTERN = "*.txt"
    
    # Class information
    CLASSES = {0: "fragment"}
    NUM_CLASSES = 1

def create_rgbd_dataset():
    """
    Create an RGBD dataset by combining RGB images with depth maps.
    Organizes files into train/val splits and saves as TIFF format.
    """
    # Get all RGB image paths
    rgb_files = sorted(list(Path(Config.RGB_SOURCE_PATH).glob(Config.RGB_PATTERN)))
    print(f"Found {len(rgb_files)} RGB images")
    
    # Create output directories
    output_dir = Path(Config.OUTPUT_BASE_PATH)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    train_img_dir = images_dir / "train"
    val_img_dir = images_dir / "val"
    train_label_dir = labels_dir / "train"
    val_label_dir = labels_dir / "val"
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process files
    train_count = 0
    val_count = 0
    skipped_count = 0
    
    for rgb_path in tqdm(rgb_files, desc="Processing images"):
        # Extract base name without extension (e.g., "001")
        base_name = rgb_path.stem
        
        # Construct corresponding depth and label paths based on naming pattern (001.jpg -> 001_depth.png -> 001.txt)
        depth_path = Path(Config.DEPTH_SOURCE_PATH) / f"{base_name}_depth.png"
        label_path = Path(Config.LABEL_SOURCE_PATH) / f"{base_name}.txt"
        
        # Skip if files don't exist
        if not depth_path.exists() or not label_path.exists():
            skipped_count += 1
            continue
        
        # Decide train/val split
        if random.random() > Config.VAL_SPLIT:
            dest_img_dir, dest_lbl_dir = train_img_dir, train_label_dir
            train_count += 1
        else:
            dest_img_dir, dest_lbl_dir = val_img_dir, val_label_dir
            val_count += 1
        
        # Read RGB image
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"Warning: Could not read {rgb_path}")
            skipped_count += 1
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Read depth image
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"Warning: Could not read {depth_path}")
            skipped_count += 1
            continue
        
        # Resize depth to match RGB dimensions if needed
        h, w = rgb_img.shape[:2]
        if depth_img.shape[0] != h or depth_img.shape[1] != w:
            depth_img = cv2.resize(depth_img, (w, h))
        
        # Create 4-channel RGBD image (HWC format)
        rgbd_img = np.dstack((rgb_img, depth_img))
        
        # Save as TIFF
        output_path = dest_img_dir / f"{base_name}.tiff"
        
        # Split into individual channels and save as multi-page TIFF
        try:
            channels = [rgbd_img[:,:,i] for i in range(4)]
            success = cv2.imwritemulti(str(output_path), channels)
            if not success:
                raise Exception("Failed to write multi-page TIFF")
        except Exception as e:
            print(f"Error saving TIFF: {e}")
            # Fallback: save as 4-channel PNG (less ideal but workable)
            cv2.imwrite(str(output_path).replace('.tiff', '.png'), rgbd_img)
        
        # Copy label file
        shutil.copy(str(label_path), str(dest_lbl_dir / f"{base_name}.txt"))
    
    print(f"Dataset created: {train_count} training images, {val_count} validation images")
    print(f"Skipped {skipped_count} images due to missing files or errors")
    
    # Create dataset YAML file
    yaml_path = output_dir / "rgbd.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"# RGBD dataset configuration\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"# Number of channels (RGBD)\n")
        f.write(f"channels: 4\n\n")
        f.write(f"# Classes\n")
        f.write(f"nc: {Config.NUM_CLASSES}\n")
        f.write(f"names: {list(Config.CLASSES.values())}\n")
    
    print(f"Dataset configuration saved to {yaml_path}")
    return output_dir

if __name__ == "__main__":
    # Run the dataset creation
    dataset_path = create_rgbd_dataset() 