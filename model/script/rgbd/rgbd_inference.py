import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def load_rgbd_image(rgb_path, depth_path):
    """
    Load and combine RGB and depth images into a 4-channel RGBD image.
    
    Args:
        rgb_path: Path to the RGB image
        depth_path: Path to the depth image
    
    Returns:
        rgbd_img: 4-channel RGBD image
    """
    # Load RGB image
    rgb_img = cv2.imread(str(rgb_path))
    if rgb_img is None:
        raise ValueError(f"Could not read RGB image at {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Load depth image
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        raise ValueError(f"Could not read depth image at {depth_path}")
    
    # Resize depth to match RGB dimensions if needed
    h, w = rgb_img.shape[:2]
    if depth_img.shape[0] != h or depth_img.shape[1] != w:
        depth_img = cv2.resize(depth_img, (w, h))
    
    # Create 4-channel RGBD image
    rgbd_img = np.dstack((rgb_img, depth_img))
    
    return rgbd_img

def perform_inference(model_path, rgb_path, depth_path):
    """
    Perform inference on an RGBD image pair.
    
    Args:
        model_path: Path to the trained YOLO model
        rgb_path: Path to the RGB image
        depth_path: Path to the depth image
    
    Returns:
        results: Inference results
        rgbd_img: 4-channel RGBD image
    """
    # Load model
    model = YOLO(model_path)
    
    # Load RGBD image
    rgbd_img = load_rgbd_image(rgb_path, depth_path)
    
    # Run inference
    results = model(rgbd_img)
    
    return results, rgbd_img

def visualize_results(results, rgbd_img, save_path=None):
    """
    Visualize inference results.
    
    Args:
        results: Inference results from YOLO model
        rgbd_img: 4-channel RGBD image
        save_path: Path to save the visualization (optional)
    """
    # Extract RGB and depth channels for visualization
    rgb_vis = rgbd_img[:,:,:3]  # Just the RGB channels
    depth_vis = rgbd_img[:,:,3]  # Depth channel
    
    plt.figure(figsize=(16, 12))
    
    # RGB input
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_vis)
    plt.title('RGB Input')
    plt.axis('off')
    
    # Depth input
    plt.subplot(2, 2, 2)
    plt.imshow(depth_vis, cmap='gray')
    plt.title('Depth Channel')
    plt.axis('off')
    
    # Segmentation result
    plt.subplot(2, 2, 3)
    plt.imshow(results[0].plot())
    plt.title('Segmentation Result')
    plt.axis('off')
    
    # Masks only
    plt.subplot(2, 2, 4)
    if hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks) > 0:
        mask = results[0].masks.data[0].cpu().numpy()
        plt.imshow(mask, cmap='jet', alpha=0.7)
        plt.imshow(rgb_vis, alpha=0.3)
        plt.title('Segmentation Mask')
    else:
        plt.imshow(rgb_vis)
        plt.title('No Mask Available')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def process_directory(model_path, rgb_dir, depth_dir, pattern="*.jpg", output_dir=None):
    """
    Process all matching image pairs in the given directories.
    
    Args:
        model_path: Path to the trained YOLO model
        rgb_dir: Directory containing RGB images
        depth_dir: Directory containing depth images
        pattern: File pattern to match
        output_dir: Directory to save visualizations (optional)
    """
    rgb_files = sorted(list(Path(rgb_dir).glob(pattern)))
    print(f"Found {len(rgb_files)} RGB images to process")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    for rgb_path in rgb_files:
        base_name = rgb_path.stem
        depth_path = Path(depth_dir) / f"{base_name}_depth.png"
        
        if not depth_path.exists():
            print(f"Skipping {base_name}: No matching depth image found")
            continue
        
        try:
            # Process image pair
            print(f"Processing {base_name}...")
            results, rgbd_img = perform_inference(model, rgb_path, depth_path)
            
            # Save visualization if requested
            if output_dir:
                save_path = Path(output_dir) / f"{base_name}_result.png"
                visualize_results(results, rgbd_img, save_path)
            else:
                visualize_results(results, rgbd_img)
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

if __name__ == "__main__":
    # Example usage
    model_path = "runs/segment/train/rgbd_model.pt"
    
    # Process a single image
    rgb_path = "sample_rgb.jpg"
    depth_path = "sample_depth.png"
    
    if Path(rgb_path).exists() and Path(depth_path).exists():
        results, rgbd_img = perform_inference(model_path, rgb_path, depth_path)
        visualize_results(results, rgbd_img, "sample_result.png")
    else:
        print("Sample images not found. Please provide valid RGB and depth images.")
    
    # Uncomment to process a directory
    """
    process_directory(
        model_path=model_path,
        rgb_dir="/path/to/rgb_images",
        depth_dir="/path/to/depth_images",
        output_dir="results"
    )
    """ 