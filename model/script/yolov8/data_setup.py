import os
import glob
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Config:
    def __init__(self):
        # Input paths
        self.image_dir = '/kaggle/input/fst-depth-map/vits'
        self.label_dir = '/kaggle/input/fst-mask-convert'  # Output from mask_converter.py
        
        # Output paths
        self.output_dir = '/kaggle/working'
        self.dataset_dir = '/kaggle/working/dataset'
        self.data_yaml_path = '/kaggle/working/dataset.yaml'
        
        # Dataset settings
        self.val_split = 0.2
        self.seed = 42
        self.class_name = 'fragment'
        
        # Processing options
        self.visualize_samples = 0  # Number of samples to visualize (0 for none)
        self.copy_strategy = 'copy'  # 'copy' or 'symlink'
        self.mode = "Depth"  # Options: None or "Depth"

def rename_image_files(image_dir):
    """Rename image files to remove '_depth' suffix."""
    temp_dir = '/kaggle/working/temp_images'  # Writable directory
    os.makedirs(temp_dir, exist_ok=True)
    
    renamed_count = 0
    for image_file in glob.glob(os.path.join(image_dir, '*_depth.png')):
        base_name = Path(image_file).stem.replace('_depth', '')
        new_file = f"{base_name}.png"
        new_path = os.path.join(temp_dir, new_file)
        shutil.copy2(image_file, new_path)
        renamed_count += 1
    
    print(f"Renamed {renamed_count} image files in {temp_dir}")
    return temp_dir

def setup_directory_structure(config):
    """Create directory structure for YOLOv8 training."""
    dirs = {
        'dataset': config.dataset_dir,
        'train_img': os.path.join(config.dataset_dir, 'train/images'),
        'train_label': os.path.join(config.dataset_dir, 'train/labels'),
        'val_img': os.path.join(config.dataset_dir, 'val/images'),
        'val_label': os.path.join(config.dataset_dir, 'val/labels'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def create_dataset_yaml(config, dirs):
    """Create YOLO dataset YAML file."""
    yaml_content = {
        'path': dirs['dataset'],
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': {0: config.class_name}
    }
    
    with open(config.data_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset YAML at {config.data_yaml_path}")
    return config.data_yaml_path

def copy_files(src_files, dst_dir, strategy='copy'):
    """Copy files to destination directory with progress bar."""
    for src in tqdm(src_files, desc=f"Copying to {os.path.basename(dst_dir)}"):
        filename = os.path.basename(src)
        dst = os.path.join(dst_dir, filename)
        
        if strategy == 'symlink' and os.name != 'nt':  # Symlinks not fully supported on Windows
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

def copy_labels_with_depth_suffix(label_dir, output_dir):
    """Copy label files and rename them with _depth suffix."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all label files
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    # Copy and rename files with _depth suffix
    for label_path in tqdm(label_files, desc="Copying and renaming depth labels"):
        filename = Path(label_path).stem
        new_filename = f"{filename}_depth.txt"
        output_path = os.path.join(output_dir, new_filename)
        
        # Copy the file with new name
        shutil.copy2(label_path, output_path)
    
    print(f"Copied {len(label_files)} label files to {output_dir} with '_depth' suffix")
    return output_dir

def setup_data(config):
    """Setup data for YOLOv8 training."""
    print("Setting up data structure for YOLOv8 training...")
    
    # Rename image files to remove '_depth' suffix if in Depth mode
    image_dir = config.image_dir
    if config.mode == "Depth":
        print("Depth mode enabled. Renaming image files to remove '_depth' suffix...")
        image_dir = rename_image_files(config.image_dir)
    
    # Create directory structure
    dirs = setup_directory_structure(config)
    
    # Get all image files
    image_paths = []
    for ext in ['png']:  # Only include png since images are *.png
        image_paths.extend(glob.glob(os.path.join(image_dir, f'*.{ext}')))
    
    # Get label paths
    label_paths = [os.path.join(config.label_dir, f"{Path(img).stem}.txt") for img in image_paths]
    
    # Filter to only include images with corresponding labels
    valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, label_paths) if os.path.exists(lbl)]
    if not valid_pairs:
        raise ValueError(f"No valid image-label pairs found! Check that {config.label_dir} contains label files.")
    
    img_paths, lbl_paths = zip(*valid_pairs)
    print(f"Found {len(img_paths)} valid image-label pairs")
    
    # Split dataset into train/val
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        img_paths, lbl_paths, test_size=config.val_split, random_state=config.seed
    )
    
    print(f"Split dataset: {len(train_imgs)} training, {len(val_imgs)} validation images")
    
    # Copy files to respective directories
    copy_files(train_imgs, dirs['train_img'], config.copy_strategy)
    copy_files(train_lbls, dirs['train_label'], config.copy_strategy)
    copy_files(val_imgs, dirs['val_img'], config.copy_strategy)
    copy_files(val_lbls, dirs['val_label'], config.copy_strategy)
    
    # Create dataset YAML file
    yaml_path = create_dataset_yaml(config, dirs)
    
    # Clean up temporary image directory
    if config.mode == "Depth":
        shutil.rmtree(image_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory {image_dir}")
    
    print("Data setup complete!")
    return yaml_path

def main():
    """Main function."""
    # Initialize configuration
    config = Config()
    
    # Setup data
    yaml_path = setup_data(config)
    
    print(f"Dataset prepared at: {yaml_path}")
    print("You can now run training with the YOLOv8 model.")

if __name__ == "__main__":
    main()