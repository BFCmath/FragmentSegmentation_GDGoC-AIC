import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_segment_outlines(image_path, output_path, output_viz_path=None):
    """
    Extract outlines from segmentation masks.
    
    Args:
        image_path (str): Path to the input segmentation mask
        output_path (str): Path to save the output outline mask
        output_viz_path (str, optional): Path to save visualization
    """
    # Read the image
    pil_image = Image.open(image_path)
    image = np.array(pil_image)
    
    # Create a blank image for outlines
    outline_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Handle RGB images (multi-class segmentation)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to a flattened array of RGB values
        original_shape = image.shape
        flattened = image.reshape(-1, original_shape[2])
        unique_colors = np.unique(flattened, axis=0)
        
        for color in unique_colors:
            # Skip black (0,0,0) which is typically background
            if np.all(color == 0):
                continue
            
            # Create a mask for this color
            color_mask = np.all(image == color.reshape(1, 1, -1), axis=2).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(outline_image, contours, -1, 255, 1)
    else:
        # Grayscale image - handle as multiple segments based on pixel values
        unique_values = np.unique(image)
        
        for value in unique_values:
            # Skip 0 (typically background)
            if value == 0:
                continue
            
            # Create mask for this value
            class_mask = (image == value).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(outline_image, contours, -1, 255, 1)
    
    # Save the outline mask
    cv2.imwrite(output_path, outline_image)
    
    # Create visualization if needed
    if output_viz_path:
        plt.figure(figsize=(15, 5))
        
        # Show original image
        plt.subplot(1, 3, 1)
        if len(image.shape) == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
        plt.title('Original Segmentation')
        plt.axis('off')
        
        # Show outline
        plt.subplot(1, 3, 2)
        plt.imshow(outline_image, cmap='gray')
        plt.title('Extracted Outline')
        plt.axis('off')
        
        # Show overlay on original
        plt.subplot(1, 3, 3)
        if len(image.shape) == 3:
            # Create RGB outline for overlay
            rgb_outline = np.zeros_like(image)
            rgb_outline[:,:,0] = outline_image  # Red channel
            
            # Combine with transparency
            overlay = image.copy()
            overlay[outline_image > 0] = [255, 0, 0]  # Red outline
            plt.imshow(overlay)
        else:
            # For grayscale
            plt.imshow(image, cmap='gray')
            plt.imshow(outline_image, cmap='Reds', alpha=0.5)
            
        plt.title('Outline Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_viz_path)
        plt.close()
    
    return outline_image

def process_directory(input_dir, output_dir, viz_dir=None, extension='.png'):
    """
    Process all masks in a directory and extract outlines.
    
    Args:
        input_dir (str): Directory containing input masks
        output_dir (str): Directory to save output outline masks
        viz_dir (str, optional): Directory to save visualizations
        extension (str): File extension to filter images
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(extension)]
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing masks"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        if viz_dir:
            viz_path = os.path.join(viz_dir, f"viz_{image_file}")
            extract_segment_outlines(input_path, output_path, viz_path)
        else:
            extract_segment_outlines(input_path, output_path)

if __name__ == "__main__":
    # Define directories
    input_dir = "train_data/masks"
    output_dir = "test_idea/shrinking/outlines"
    viz_dir = "test_idea/shrinking/outline_visualizations"
    
    # Process all images in the directory
    process_directory(input_dir, output_dir, viz_dir, extension='.png')
    
    print(f"Finished extracting outlines from {input_dir}")
    print(f"Outlines saved to {output_dir}")
    print(f"Visualizations saved to {viz_dir}")
