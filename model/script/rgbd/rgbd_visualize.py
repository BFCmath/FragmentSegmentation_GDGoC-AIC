import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2
import time
from pathlib import Path
from rgbd_inference import load_rgbd_image

# Set random seed for reproducibility
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)

class Config:
    def __init__(self):
        # Model settings
        self.model_path = '/kaggle/working/rgbd_model.pt'  # Path to trained model
        
        # Inference settings
        self.conf_threshold = 0.25  # Confidence threshold
        self.iou_threshold = 0.7    # IoU threshold for NMS
        self.device = 'cpu'         # Force CPU usage
        
        # --- Settings for Separate Inference Runs Timing ---
        self.rgb_image_dir_path = '/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images'
        self.depth_image_dir_path = '/kaggle/input/fst-depth-map/vits'
        self.num_separate_runs_for_timing = 50 # Number of separate inference runs on different images

        # --- Output Settings for the Single Visualized Example ---
        self.output_path_base = '/kaggle/working/rgbd_inference_eval_outputs' 
        
        # Inference/Visualization settings for YOLO predict calls
        self.imgsz = 512            # Same size used during training
        self.show_labels = False    # Set to True to show labels
        self.show_conf = False      # Set to True to show confidences
        self.show_boxes = False     # Set to True to show bounding boxes
        
        # Class information
        self.classes = ['fragment']  # Single class: fragment

def load_model(config):
    """Load the trained YOLOv8 model"""
    try:
        model = YOLO(config.model_path)
        print(f"Model loaded successfully from {config.model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_random_image_pairs(rgb_dir_str, depth_dir_str, num_pairs, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """Gets a list of random RGB-Depth image pairs."""
    rgb_dir = Path(rgb_dir_str)
    depth_dir = Path(depth_dir_str)
    
    if not rgb_dir.is_dir():
        print(f"Error: Provided RGB image path '{rgb_dir_str}' is not a directory.")
        return []
    if not depth_dir.is_dir():
        print(f"Error: Provided depth image path '{depth_dir_str}' is not a directory.")
        return []

    # Get all RGB image files
    all_rgb_files = []
    for ext in extensions:
        all_rgb_files.extend(list(rgb_dir.glob(f'*{ext}')))
    
    all_rgb_files = [str(p) for p in all_rgb_files] 

    if not all_rgb_files:
        print(f"No RGB images found in '{rgb_dir_str}' with extensions {extensions}")
        return []

    # Filter to include only images that have matching depth maps
    matched_pairs = []
    for rgb_path in all_rgb_files:
        base_name = Path(rgb_path).stem
        depth_path = str(depth_dir / f"{base_name}_depth.png")
        if os.path.exists(depth_path):
            matched_pairs.append((rgb_path, depth_path))
    
    if not matched_pairs:
        print(f"No matching RGB-Depth pairs found.")
        return []

    if len(matched_pairs) < num_pairs:
        print(f"Warning: Requested {num_pairs} pairs, but only found {len(matched_pairs)}. Using all {len(matched_pairs)} found pairs.")
        if not matched_pairs: 
            return []
        return random.choices(matched_pairs, k=num_pairs)
    else:
        return random.sample(matched_pairs, num_pairs)

def evaluate_separate_inference_runs(model, image_pairs, config):
    """Runs a separate model.predict() call for each RGBD image pair and collects timing."""
    if not model:
        print("No model loaded. Cannot run inference for timing evaluation.")
        return None

    inference_times = []
    
    print(f"\nEvaluating inference time with {len(image_pairs)} separate runs...")
    
    if image_pairs:
        print("Performing a warm-up inference run...")
        valid_warmup_pair = next((p for p in image_pairs if os.path.exists(p[0]) and os.path.exists(p[1])), None)
        if valid_warmup_pair:
            try:
                rgb_path, depth_path = valid_warmup_pair
                # Load RGBD image
                rgbd_img = load_rgbd_image(rgb_path, depth_path)
                
                # Run warm-up inference
                _ = model.predict(
                    source=rgbd_img,  # Pass RGBD numpy array directly
                    conf=config.conf_threshold, 
                    iou=config.iou_threshold, 
                    imgsz=config.imgsz,
                    device=config.device, 
                    save=False, 
                    visualize=False, 
                    verbose=False,
                    retina_masks=True, 
                    show_boxes=config.show_boxes
                )
                print("Warm-up complete.")
            except Exception as e:
                print(f"Warm-up run failed for {valid_warmup_pair}: {e}")
        else:
            print("Warm-up skipped: No valid image pairs found in the provided list for warm-up.")

    for i, (rgb_path, depth_path) in enumerate(image_pairs):
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"RGB or depth image not found at {rgb_path} or {depth_path}, skipping run {i+1}.")
            continue

        try:
            # Load RGBD image
            rgbd_img = load_rgbd_image(rgb_path, depth_path)
            
            # Run inference and time it
            start_time = time.time()
            _ = model.predict(
                source=rgbd_img,  # Pass RGBD numpy array directly
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                imgsz=config.imgsz,
                device=config.device,
                save=False, 
                save_txt=False, 
                project=None, 
                name=None,
                visualize=False, 
                show_labels=config.show_labels,
                show_conf=config.show_conf,
                show_boxes=config.show_boxes,
                retina_masks=True, 
                verbose=False 
            )
            end_time = time.time()
            current_inference_time = end_time - start_time
            inference_times.append(current_inference_time)
            if (i + 1) % 10 == 0 or i == len(image_pairs) - 1:
                 print(f"Completed run {i+1}/{len(image_pairs)} on {Path(rgb_path).name}/{Path(depth_path).name} in {current_inference_time:.4f}s")
        except Exception as e:
            print(f"Error during inference for {rgb_path}/{depth_path} (run {i+1}): {e}")
            
    if not inference_times:
        print("No images were successfully processed for timing.")
        return []
        
    return inference_times

def visualize_rgbd_result(result, rgb_path, depth_path, inference_time, output_visualization_path, config):
    """Visualize RGBD image prediction with side-by-side comparison.
       Saves the custom plot to output_visualization_path.
    """
    if result is None:
        print("No result to visualize")
        return
    
    os.makedirs(Path(output_visualization_path).parent, exist_ok=True)
    
    # Load RGBD image
    rgbd_img = load_rgbd_image(rgb_path, depth_path)
    rgb_img = rgbd_img[:,:,:3]  # Extract RGB channels
    depth_img = rgbd_img[:,:,3]  # Extract depth channel
    
    plt.figure(figsize=(20, 10))
    
    # Original RGB image
    plt.subplot(2, 3, 1)
    try:
        plt.imshow(rgb_img)
        plt.title(f"RGB Image: {Path(rgb_path).name}")
    except Exception as e:
        plt.title(f"Could not load RGB: {Path(rgb_path).name}\n{e}")
    plt.axis('off')
    
    # Depth image
    plt.subplot(2, 3, 2)
    try:
        plt.imshow(depth_img, cmap='gray')
        plt.title(f"Depth Channel: {Path(depth_path).name}")
    except Exception as e:
        plt.title(f"Could not load depth: {Path(depth_path).name}\n{e}")
    plt.axis('off')
    
    # Combined RGBD visualization (RGB with depth as alpha)
    plt.subplot(2, 3, 3)
    try:
        # Normalize depth for visualization
        depth_normalized = depth_img / (np.max(depth_img) + 1e-10)
        plt.imshow(rgb_img)
        plt.imshow(depth_normalized, cmap='inferno', alpha=0.4)
        plt.title("RGBD Combined")
    except Exception as e:
        plt.title(f"Could not create RGBD visualization\n{e}")
    plt.axis('off')
    
    # Segmentation masks
    plt.subplot(2, 3, 4)
    if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
        masks_data = result.masks.data.cpu().numpy()
        mask_img = np.zeros_like(rgb_img)
        
        for j, mask in enumerate(masks_data):
            color = np.array([random.randint(50, 255), random.randint(50, 200), random.randint(50, 200)])
            mask_h, mask_w = mask.shape
            img_h, img_w = rgb_img.shape[:2]
            bin_mask = mask.astype('uint8')
            if mask_h != img_h or mask_w != img_w:
                bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_img[bin_mask > 0] = color
        
        alpha = 0.6
        blended = cv2.addWeighted(rgb_img, 1 - alpha, mask_img, alpha, 0)
        plt.imshow(blended)
        plt.title(f"Masks ({len(masks_data)} objects)")
    else:
        plt.imshow(rgb_img)
        plt.title("No masks detected")
    plt.axis('off')
    
    # Default YOLO visualization
    plt.subplot(2, 3, 5)
    plt.imshow(result.plot(
        show_boxes=config.show_boxes, 
        show_labels=config.show_labels, 
        show_conf=config.show_conf
    ))
    plt.title(f"YOLO Prediction ({inference_time:.3f}s)")
    plt.axis('off')
    
    # Depth with mask overlay
    plt.subplot(2, 3, 6)
    if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
        # Create a colormap for the depth
        depth_colored = plt.cm.viridis(depth_normalized)[:, :, :3]  # Convert depth to RGB using colormap
        
        # Create an overlay of masks
        mask_overlay = np.zeros_like(depth_colored)
        for j, mask in enumerate(masks_data):
            mask_h, mask_w = mask.shape
            img_h, img_w = rgb_img.shape[:2]
            bin_mask = mask.astype('uint8')
            if mask_h != img_h or mask_w != img_w:
                bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            
            # Create a colored mask
            color = np.array([random.random(), random.random(), random.random()])
            for c in range(3):
                mask_overlay[:, :, c] = np.where(bin_mask > 0, color[c], mask_overlay[:, :, c])
        
        # Blend depth with mask overlay
        depth_blend = 0.7 * depth_colored + 0.3 * mask_overlay
        plt.imshow(depth_blend)
        plt.title("Depth with Mask Overlay")
    else:
        plt.imshow(depth_normalized, cmap='viridis')
        plt.title("Depth (No masks detected)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_visualization_path)
    print(f"Custom RGBD visualization saved to {output_visualization_path}")
    plt.show()
    plt.close()

def main():
    """Main function"""
    config = Config() 
    
    model = load_model(config)
    if not model:
        return

    image_pairs = get_random_image_pairs(
        config.rgb_image_dir_path, 
        config.depth_image_dir_path, 
        config.num_separate_runs_for_timing
    )
    
    if not image_pairs:
        print(f"No image pairs found for evaluation. Exiting timing evaluation.")
    else:
        all_separate_inference_times = evaluate_separate_inference_runs(model, image_pairs, config)

        if all_separate_inference_times:
            valid_times = [t for t in all_separate_inference_times if not np.isnan(t)]
            if not valid_times:
                print("No valid inference times recorded.")
            else:
                total_time_all_runs = sum(valid_times)
                avg_time = np.mean(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                std_dev_time = np.std(valid_times)
                median_time = np.median(valid_times)
                
                print("\n--- RGBD Separate Inference Runs Timing Summary ---")
                print(f"Number of runs successfully timed: {len(valid_times)} / {len(image_pairs)}")
                print(f"Total time for all {len(valid_times)} separate runs: {total_time_all_runs:.4f} seconds")
                print(f"Average inference time per run: {avg_time:.4f} seconds")
                print(f"Median inference time per run: {median_time:.4f} seconds")
                print(f"Minimum inference time per run: {min_time:.4f} seconds")
                print(f"Maximum inference time per run: {max_time:.4f} seconds")
                print(f"Standard deviation of inference time: {std_dev_time:.4f} seconds")
        else:
            print("Separate inference runs timing did not produce any results.")

        if image_pairs:
            print("\n--- Running and Visualizing One RGBD Example ---")
            
            example_pair = random.choice(image_pairs)
            rgb_path, depth_path = example_pair
            
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                print(f"Selected example images {rgb_path} or {depth_path} not found. Skipping visualization.")
            else:
                example_specific_output_folder_name = f"visualized_rgbd_example_{Path(rgb_path).stem}"
                yolo_project_dir = Path(config.output_path_base)
                yolo_run_name = example_specific_output_folder_name 
                
                os.makedirs(yolo_project_dir / yolo_run_name, exist_ok=True)
                
                print(f"Running inference for visualization on: {rgb_path} and {depth_path}")
                
                try:
                    # Load RGBD image
                    rgbd_img = load_rgbd_image(rgb_path, depth_path)
                    
                    start_time_single_viz = time.time()
                    results_single_list = model.predict(
                        source=rgbd_img,
                        conf=config.conf_threshold, 
                        iou=config.iou_threshold, 
                        imgsz=config.imgsz,
                        device=config.device,
                        save=True,          
                        save_txt=False,
                        project=str(yolo_project_dir), 
                        name=yolo_run_name,       
                        visualize=False,    
                        show_labels=config.show_labels,
                        show_conf=config.show_conf,
                        show_boxes=config.show_boxes,
                        retina_masks=True, 
                        verbose=True 
                    )
                    end_time_single_viz = time.time()
                    inference_time_single_viz = end_time_single_viz - start_time_single_viz
                    print(f"Single RGBD example inference (for visualization) completed in {inference_time_single_viz:.4f} seconds")

                    if results_single_list:
                        result_single = results_single_list[0]
                        custom_plot_filename = f"custom_rgbd_visualization_{Path(rgb_path).stem}.png"
                        # Ensure result_single.save_dir is valid, otherwise use a fallback
                        save_dir_path = Path(result_single.save_dir) if hasattr(result_single, 'save_dir') and result_single.save_dir else yolo_project_dir / yolo_run_name / "predict" # Fallback if save_dir is None
                        if not save_dir_path.exists():
                            save_dir_path.mkdir(parents=True, exist_ok=True) # Create if it doesn't exist
                        custom_plot_save_path = save_dir_path / custom_plot_filename
                        
                        visualize_rgbd_result(result_single, rgb_path, depth_path, inference_time_single_viz, str(custom_plot_save_path), config)
                    else:
                        print("No results returned for the single RGBD example visualization.")
                except Exception as e:
                    print(f"Error during single RGBD example inference/visualization for {rgb_path} and {depth_path}: {e}")
        else:
            print("\nSkipping single example visualization as no image pairs were selected for timing runs.")

    print("\nRGBD evaluation script finished.")

if __name__ == '__main__':
    main() 