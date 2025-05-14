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

# Set random seed for reproducibility
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)

class Config:
    def __init__(self):
        # Model settings
        self.model_path = '/kaggle/working/best.pt'  # Path to trained model
        
        # Inference settings
        self.conf_threshold = 0.25  # Confidence threshold
        self.iou_threshold = 0.7   # IoU threshold for NMS
        self.device = 'cpu'         # Force CPU usage
        
        # --- Settings for Separate Inference Runs Timing ---
        self.image_dir_path = '/kaggle/input/fst-depth-map/vits/' 
        self.num_separate_runs_for_timing = 50 # Number of separate inference runs on different images

        # --- Output Settings for the Single Visualized Example ---
        self.output_path_base = '/kaggle/working/inference_eval_outputs' 
        
        # Inference/Visualization settings for YOLO predict calls
        self.imgsz = 512            # Same size used during training
        # Updated based on deprecation warnings:
        self.show_labels = False    # Set to True to show labels (previously hide_labels=True)
        self.show_conf = False      # Set to True to show confidences (previously hide_conf=True)
        self.show_boxes = False     # Set to True to show bounding boxes (previously boxes=False for predict)
        
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

def get_random_image_paths(image_dir_str, num_images, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """Gets a list of random image paths from a directory."""
    image_dir = Path(image_dir_str)
    if not image_dir.is_dir():
        print(f"Error: Provided image path '{image_dir_str}' is not a directory.")
        return []

    all_image_files = []
    for ext in extensions:
        all_image_files.extend(list(image_dir.rglob(f'*{ext}')))
    
    all_image_files = [str(p) for p in all_image_files] 

    if not all_image_files:
        print(f"No images found in '{image_dir_str}' with extensions {extensions}")
        return []

    if len(all_image_files) < num_images:
        print(f"Warning: Requested {num_images} images, but only found {len(all_image_files)}. Using all {len(all_image_files)} found images.")
        if not all_image_files: return []
        return random.choices(all_image_files, k=num_images)
    else:
        return random.sample(all_image_files, num_images)


def evaluate_separate_inference_runs(model, image_paths, config):
    """Runs a separate model.predict() call for each image and collects timing."""
    if not model:
        print("No model loaded. Cannot run inference for timing evaluation.")
        return None

    inference_times = []
    
    print(f"\nEvaluating inference time with {len(image_paths)} separate runs...")
    
    if image_paths:
        print("Performing a warm-up inference run...")
        valid_warmup_path = next((p for p in image_paths if os.path.exists(p)), None)
        if valid_warmup_path:
            try:
                _ = model.predict(
                    source=valid_warmup_path,
                    conf=config.conf_threshold, iou=config.iou_threshold, imgsz=config.imgsz,
                    device=config.device, save=False, visualize=False, verbose=False,
                    retina_masks=True, 
                    show_boxes=config.show_boxes # Updated
                )
                print("Warm-up complete.")
            except Exception as e:
                print(f"Warm-up run failed for {valid_warmup_path}: {e}")
        else:
            print("Warm-up skipped: No valid image path found in the provided list for warm-up.")


    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Image not found at {image_path}, skipping run {i+1}.")
            continue

        try:
            start_time = time.time()
            _ = model.predict(
                source=image_path, 
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                imgsz=config.imgsz,
                device=config.device,
                save=False, save_txt=False, project=None, name=None,
                visualize=False, 
                show_labels=config.show_labels, # Updated
                show_conf=config.show_conf,     # Updated
                show_boxes=config.show_boxes,   # Updated
                retina_masks=True, verbose=False 
            )
            end_time = time.time()
            current_inference_time = end_time - start_time
            inference_times.append(current_inference_time)
            if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
                 print(f"Completed run {i+1}/{len(image_paths)} on {Path(image_path).name} in {current_inference_time:.4f}s")
        except Exception as e:
            print(f"Error during inference for {image_path} (run {i+1}): {e}")
            
    if not inference_times:
        print("No images were successfully processed for timing.")
        return []
        
    return inference_times

def visualize_single_result(result, img_path, inference_time, output_visualization_path, config): # Added config
    """Visualize single image prediction with side-by-side comparison.
       Saves the custom plot to output_visualization_path.
    """
    if result is None:
        print("No result to visualize")
        return
    
    os.makedirs(Path(output_visualization_path).parent, exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    try:
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.title(f"Original Image: {Path(img_path).name}")
    except Exception as e:
        plt.title(f"Could not load original: {Path(img_path).name}\n{e}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            masks_data = result.masks.data.cpu().numpy()
            mask_img = np.zeros_like(orig_img)
            
            for j, mask in enumerate(masks_data):
                color = np.array([random.randint(50, 255), random.randint(50, 200), random.randint(50, 200)])
                mask_h, mask_w = mask.shape
                img_h, img_w = orig_img.shape[:2]
                bin_mask = mask.astype('uint8')
                if mask_h != img_h or mask_w != img_w:
                    bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                mask_img[bin_mask > 0] = color
            
            alpha = 0.5
            blended = cv2.addWeighted(orig_img, 1 - alpha, mask_img, alpha, 0)
            plt.imshow(blended)
            plt.title(f"Prediction ({len(masks_data)} objects, {inference_time:.3f}s)")
        else:
            # Use result.plot for fallback if CV2 fails
            plt.imshow(result.plot(
                show_boxes=config.show_boxes, 
                show_labels=config.show_labels, 
                show_conf=config.show_conf
            ))
            plt.title(f"Prediction (CV2 Load Fail, {inference_time:.3f}s)")
    else:
        # Use result.plot if no masks or for default plotting
        plt.imshow(result.plot(
            show_boxes=config.show_boxes, 
            show_labels=config.show_labels, 
            show_conf=config.show_conf
        ))
        plt.title(f"Prediction (No masks or default plot, {inference_time:.3f}s)")
    
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_visualization_path)
    print(f"Custom visualization saved to {output_visualization_path}")
    plt.show()
    plt.close()

def main():
    """Main function"""
    config = Config() 
    
    model = load_model(config)
    if not model:
        return

    image_paths_for_eval = get_random_image_paths(config.image_dir_path, config.num_separate_runs_for_timing)
    
    if not image_paths_for_eval:
        print(f"No images found in {config.image_dir_path} for evaluation. Exiting timing evaluation.")
    else:
        all_separate_inference_times = evaluate_separate_inference_runs(model, image_paths_for_eval, config)

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
                
                print("\n--- Separate Inference Runs Timing Summary ---")
                print(f"Number of runs successfully timed: {len(valid_times)} / {len(image_paths_for_eval)}")
                print(f"Total time for all {len(valid_times)} separate runs: {total_time_all_runs:.4f} seconds")
                print(f"Average inference time per run: {avg_time:.4f} seconds")
                print(f"Median inference time per run: {median_time:.4f} seconds")
                print(f"Minimum inference time per run: {min_time:.4f} seconds")
                print(f"Maximum inference time per run: {max_time:.4f} seconds")
                print(f"Standard deviation of inference time: {std_dev_time:.4f} seconds")
        else:
            print("Separate inference runs timing did not produce any results.")

        if image_paths_for_eval:
            print("\n--- Running and Visualizing One Example ---")
            
            example_image_path = random.choice(image_paths_for_eval) 
            if not os.path.exists(example_image_path):
                print(f"Selected example image {example_image_path} not found. Skipping visualization.")
                # Optionally, try another image or exit
            else:
                example_specific_output_folder_name = f"visualized_example_{Path(example_image_path).stem}"
                yolo_project_dir = Path(config.output_path_base)
                yolo_run_name = example_specific_output_folder_name 
                
                os.makedirs(yolo_project_dir / yolo_run_name, exist_ok=True)
                
                print(f"Running inference for visualization on: {example_image_path}")
                start_time_single_viz = time.time()
                try:
                    results_single_list = model.predict(
                        source=example_image_path,
                        conf=config.conf_threshold, iou=config.iou_threshold, imgsz=config.imgsz,
                        device=config.device,
                        save=True,          
                        save_txt=False,
                        project=str(yolo_project_dir), 
                        name=yolo_run_name,       
                        visualize=False,    
                        show_labels=config.show_labels, # Updated
                        show_conf=config.show_conf,     # Updated
                        show_boxes=config.show_boxes,   # Updated
                        retina_masks=True, verbose=True 
                    )
                    end_time_single_viz = time.time()
                    inference_time_single_viz = end_time_single_viz - start_time_single_viz
                    print(f"Single example inference (for visualization) completed in {inference_time_single_viz:.4f} seconds")

                    if results_single_list:
                        result_single = results_single_list[0]
                        custom_plot_filename = f"custom_visualization_{Path(example_image_path).stem}.png"
                        # Ensure result_single.save_dir is valid, otherwise use a fallback
                        save_dir_path = Path(result_single.save_dir) if hasattr(result_single, 'save_dir') and result_single.save_dir else yolo_project_dir / yolo_run_name / "predict" # Fallback if save_dir is None
                        if not save_dir_path.exists():
                            save_dir_path.mkdir(parents=True, exist_ok=True) # Create if it doesn't exist
                        custom_plot_save_path = save_dir_path / custom_plot_filename
                        
                        visualize_single_result(result_single, example_image_path, inference_time_single_viz, str(custom_plot_save_path), config) # Pass config
                    else:
                        print("No results returned for the single example visualization.")
                except Exception as e:
                    print(f"Error during single example inference/visualization for {example_image_path}: {e}")
        else:
            print("\nSkipping single example visualization as no images were selected for timing runs.")

    print("\nEvaluation script finished.")

if __name__ == '__main__':
    main()