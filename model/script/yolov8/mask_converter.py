import cv2
import numpy as np
import os
import glob
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        # Input/Output paths
        self.mode = "mask2yolo"  # Options: "mask2yolo", "yolo2mask"
        self.input_dir = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/masks"
        self.output_dir = "/kaggle/working"
        self.image_dir = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images"
        self.class_id = 0  # Starting class ID for mask2yolo mode
        self.width = 512  # Image width for yolo2mask mode
        self.height = 512  # Image height for yolo2mask mode
        self.sanity_check_interval = 50  # Run sanity check after every 5 images
        self.sanity_check_num_visualizations = 3  # Visualize up to 3 images per check

def find_connected_components(binary_mask):
    """Find connected components in a binary mask."""
    labeled_mask, num_components = ndimage.label(binary_mask)
    return labeled_mask, num_components

def get_largest_component(labeled_mask, component_id):
    """Get the largest connected component from a labeled mask."""
    return (labeled_mask == component_id).astype(np.uint8)

def mask_to_yolo(mask_path, output_dir, class_id=0):
    """Convert a segmentation mask to YOLO format with consistent class ID for all objects."""
    os.makedirs(output_dir, exist_ok=True)
    
    mask = cv2.imread(mask_path)
    height, width = mask.shape[:2]
    
    filename = Path(mask_path).stem
    output_path = os.path.join(output_dir, f"{filename}.txt")
    all_polygons = []
    
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        pixels = mask.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        for color in unique_colors:
            if np.all(color <= [15, 15, 15]):  # Skip background
                continue
            color_mask = np.all(mask == color.reshape(1, 1, 3), axis=2).astype(np.uint8)
            
            # Find connected components for this color
            labeled_mask, num_components = find_connected_components(color_mask)
            
            for component_id in range(1, num_components + 1):
                component_mask = get_largest_component(labeled_mask, component_id)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                normalized_points = approx_polygon.reshape(-1, 2).astype(float)
                normalized_points[:, 0] /= width
                normalized_points[:, 1] /= height
                
                # Always use the original class_id parameter for all polygons
                all_polygons.append((normalized_points, class_id))
        
        with open(output_path, 'w') as f:
            for points, cid in all_polygons:
                yolo_line = f"{cid} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                f.write(yolo_line + "\n")
    
    return output_path

def yolo_to_mask(yolo_path, img_size, image_dir=None):
    """Convert YOLO format segmentation to a mask image."""
    width, height = img_size
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    filename = Path(yolo_path).stem
    if image_dir:
        img_path = os.path.join(image_dir, f"{filename}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                width, height = img.shape[1], img.shape[0]
                mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    with open(yolo_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        
        coords = np.array([float(coord) for coord in parts[1:]]).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        coords = coords.astype(np.int32)
        
        # Use class_id to generate a consistent color
        np.random.seed(class_id)  # Seed with class_id for consistency
        color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
        
        cv2.fillPoly(mask, [coords], color)
    
    return mask

def visualize_sanity_check(img, orig_mask, recon_mask):
    """Visualize original image, original mask, and reconstructed mask using matplotlib."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, ::-1])  # Convert BGR to RGB
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(orig_mask)  # Display original mask as-is (RGB)
    plt.title("Original Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(recon_mask)  # Display reconstructed mask (RGB)
    plt.title("Reconstructed Mask (YOLO)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def sanity_check(config, processed_images, processed_labels):
    """Perform sanity check by visualizing original and reconstructed masks inline."""
    print(f"Running sanity check after processing {len(processed_images)} images")
    
    for i, (img_path, label_path) in enumerate(zip(processed_images, processed_labels)):
        if i >= config.sanity_check_num_visualizations:
            break
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        mask_path = os.path.join(config.input_dir, f"{img_stem}.png")
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        
        img_size = (img.shape[1], img.shape[0])
        recon_mask = yolo_to_mask(label_path, img_size, config.image_dir)
        
        visualize_sanity_check(img, mask, recon_mask)
    
    print("Sanity check visualization complete")

def batch_convert(config):
    """Batch convert masks to YOLO format with sanity checks."""
    if config.mode != "mask2yolo":
        print("Sanity check is only supported in mask2yolo mode.")
        return
    
    processed_images = []
    processed_labels = []
    
    mask_paths = glob.glob(os.path.join(config.input_dir, "*.png"))
    for mask_path in mask_paths:
        img_name = os.path.basename(mask_path)
        img_stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(config.image_dir, f"{img_stem}.jpg")
        
        if not os.path.exists(img_path):
            print(f"No image found for {img_name}")
            continue
        
        print(f"Converting {mask_path} to YOLO format...")
        label_path = mask_to_yolo(mask_path, config.output_dir, config.class_id)
        
        processed_images.append(img_path)
        processed_labels.append(label_path)
        
        if len(processed_images) % config.sanity_check_interval == 0:
            sanity_check(config, processed_images[-config.sanity_check_interval:], 
                        processed_labels[-config.sanity_check_interval:])
    
    if processed_images:
        sanity_check(config, processed_images, processed_labels)

def main():
    """Main function."""
    config = Config()
    batch_convert(config)
    print(f"Conversion complete. Output saved to {config.output_dir}")

if __name__ == "__main__":
    main()