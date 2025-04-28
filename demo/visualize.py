import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import argparse # Removed argparse import
import os

def load_json_prediction(json_path):
    """Load prediction JSON file"""
    with open(json_path, 'r') as f:
        prediction = json.load(f)
    return prediction

def generate_masks_from_prediction(prediction, output_dir=None, show_plot=True):
    """Generate and visualize masks from prediction data"""
    # Extract data
    individual_masks = prediction.get('masks', [])
    combined_mask_data = prediction.get('combined_mask', [])
    scores = prediction.get('scores', [])
    boxes = prediction.get('boxes', [])

    if not combined_mask_data and not individual_masks:
        print("No mask data found in prediction JSON")
        return

    # --- Visualization Logic ---
    # Determine if we need to create plots based on available data and show_plot flag
    plot_combined = bool(combined_mask_data) and show_plot
    plot_individual = bool(individual_masks) and show_plot
    plot_boxes = bool(boxes) and show_plot and output_dir # Only plot boxes if saving and requested

    # --- Create Figure for Masks (if plotting needed) ---
    if plot_combined or plot_individual:
        num_subplots = sum([plot_combined, plot_individual])
        if num_subplots > 0:
            fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))
            if num_subplots == 1:
                axes = [axes] # Make it iterable even with one subplot
            current_ax_idx = 0
        else:
             # If not plotting but need dimensions for saving boxes later
            if combined_mask_data:
                h, w = np.array(combined_mask_data).shape
            elif individual_masks:
                 h, w = np.array(individual_masks[0]).shape
            else:
                print("Cannot determine mask dimensions and not plotting.")
                return # Cannot proceed without dimensions if boxes need saving

    # --- Process and Plot/Save Combined Mask ---
    if combined_mask_data:
        combined_mask = np.array(combined_mask_data, dtype=np.uint8)
        if plot_combined:
            ax = axes[current_ax_idx]
            ax.imshow(combined_mask, cmap='gray')
            ax.set_title("Combined Mask")
            ax.axis('off')
            current_ax_idx += 1
            h, w = combined_mask.shape # Get dimensions

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            combined_mask_img = Image.fromarray(combined_mask * 255)  # Scale to 0-255
            save_path = os.path.join(output_dir, 'combined_mask.png')
            combined_mask_img.save(save_path)
            print(f"Saved combined mask to {save_path}")

    # --- Process and Plot/Save Individual Masks ---
    if individual_masks:
        # Determine dimensions if not already set
        if 'h' not in locals():
             h, w = np.array(individual_masks[0]).shape

        # Create a colorful visualization of individual masks
        color_mask_viz = np.zeros((h, w, 3), dtype=np.float32)
        has_saved_individual = False # Track if any individual masks were saved

        for i, mask_data in enumerate(individual_masks):
            mask = np.array(mask_data, dtype=np.uint8)
            color = np.random.rand(3) * 0.8 + 0.2 # Brighter colors
            for c in range(3):
                color_mask_viz[:, :, c] = np.where(mask == 1, color[c], color_mask_viz[:, :, c])

            # Save individual mask if requested
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                indiv_mask_img = Image.fromarray(mask * 255)
                filename = f'mask_{i+1}'
                # If we have scores, save them in the filename
                if i < len(scores):
                    score = scores[i]
                    filename += f'_score_{score:.2f}'
                filename += '.png'
                save_path = os.path.join(output_dir, filename)
                indiv_mask_img.save(save_path)
                if not has_saved_individual:
                     print(f"Saved individual masks to {output_dir}")
                     has_saved_individual = True # Print only once

        # Show colorful mask if requested
        if plot_individual:
            ax = axes[current_ax_idx]
            ax.imshow(color_mask_viz)
            ax.set_title("Individual Masks (Colored)")
            ax.axis('off')
            current_ax_idx += 1

        # Save colorful mask visualization
        if output_dir:
            color_mask_img = Image.fromarray((color_mask_viz * 255).astype(np.uint8))
            save_path = os.path.join(output_dir, 'colored_masks.png')
            color_mask_img.save(save_path)
            print(f"Saved colored mask visualization to {save_path}")

    # --- Finalize Mask Plot ---
    if plot_combined or plot_individual:
        fig.tight_layout()
        # Don't call plt.show() yet if we need to plot boxes separately

    # --- Plot/Save Bounding Boxes (if requested and data available) ---
    if boxes and output_dir:
         # Determine dimensions if not already set (should be set by now if masks existed)
        if 'h' not in locals():
             print("Cannot plot boxes without mask dimensions.")
             return

        # Create a background for the box plot
        # Prefer combined mask, fallback to black if not saved/available
        background = np.zeros((h, w, 3), dtype=np.uint8)
        combined_mask_path = os.path.join(output_dir, 'combined_mask.png')
        if os.path.exists(combined_mask_path):
            try:
                img = Image.open(combined_mask_path)
                background = np.array(img.convert('RGB'))
            except Exception as e:
                print(f"Warning: Could not load combined mask for box background: {e}")

        # Create a new figure specifically for boxes
        fig_boxes, ax_boxes = plt.subplots(figsize=(8, 8)) # Use a separate figure
        ax_boxes.imshow(background)

        # Draw boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box) # Ensure integer coordinates
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=1.5, edgecolor='r', facecolor='none')
            ax_boxes.add_patch(rect)

            # Add score if available
            if i < len(scores):
                ax_boxes.text(x1 + 2, y1 + 10, f"{scores[i]:.2f}", color='white',
                         bbox=dict(facecolor='red', alpha=0.6, pad=1, edgecolor='none'),
                         fontsize=8)

        ax_boxes.set_title("Bounding Boxes")
        ax_boxes.axis('off')
        fig_boxes.tight_layout()

        # Save the box visualization
        save_path = os.path.join(output_dir, 'bounding_boxes.png')
        fig_boxes.savefig(save_path)
        print(f"Saved bounding box visualization to {save_path}")

        # Show box plot if requested (and masks weren't already shown)
        if show_plot and not (plot_combined or plot_individual):
             plt.show(fig_boxes) # Show only the box figure
        elif plot_boxes: # If boxes are plotted AND other plots exist
            pass # They will be shown later by the final plt.show()

    # --- Show Plots if Requested ---
    if show_plot and (plot_combined or plot_individual or plot_boxes):
        plt.show() # Shows all figures created that haven't been explicitly closed

def main():
    # --- Configuration ---
    # Set the path to your JSON file here
    json_input_path = 'a.json'

    # Set the directory to save output images (e.g., 'output_masks')
    # Set to None to disable saving images
    output_directory = 'output_results'

    # Set to True to display matplotlib plots, False to run silently (saving only if output_dir is set)
    display_plots = True
    # --- End Configuration ---


    # Basic check if input file exists
    if not os.path.exists(json_input_path):
        print(f"Error: Input JSON file not found at '{json_input_path}'")
        return

    print(f"Loading predictions from: {json_input_path}")
    if output_directory:
        print(f"Saving outputs to: {output_directory}")
    if display_plots:
        print("Plot display enabled.")
    else:
        print("Plot display disabled.")


    # Load and process predictions
    try:
        predictions = load_json_prediction(json_input_path)
        generate_masks_from_prediction(
            predictions,
            output_dir=output_directory,
            show_plot=display_plots
        )
        print("Processing complete.")
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{json_input_path}'")
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from '{json_input_path}'. Is it a valid JSON file?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()