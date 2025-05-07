# YOLOv8 Instance Segmentation for Fragment Detection

This repository contains scripts to train a YOLOv8 model for instance segmentation of fragments. The workflow is divided into three independent scripts:

1. `mask_converter.py` - Converts colorful mask images to YOLO segmentation format
2. `data_setup.py` - Sets up the dataset structure for YOLOv8 training
3. `main.py` - Trains a YOLOv8 segmentation model

## Prerequisites

Install the required packages:

```bash
pip install ultralytics numpy matplotlib scikit-learn tqdm pyyaml opencv-python scikit-image
```

## Configuration

Each script has its own `Config` class that can be modified to customize the behavior:

- Input and output paths
- Training parameters
- Processing options

## Step 1: Convert Masks to YOLO Format

Run the mask converter script to convert colorful masks to YOLO segmentation format:

```bash
python mask_converter.py
```

### Configuration Options:

Edit the `Config` class in `mask_converter.py` to modify:

```python
class Config:
    def __init__(self):
        # Input/Output paths
        self.mode = "mask2yolo"  # Options: "mask2yolo", "yolo2mask"
        self.input_dir = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/masks"
        self.output_dir = "/kaggle/working"
        self.image_dir = "/kaggle/input/gd-go-c-hcmus-aic-fragment-segmentation-track/train/images"
        self.class_id = 0  # Class ID for all objects
        self.width = 512  # Image width for yolo2mask mode
        self.height = 512  # Image height for yolo2mask mode
        self.sanity_check_interval = 50  # Run sanity check after every 50 images
        self.sanity_check_num_visualizations = 3  # Visualize up to 3 images per check
```

The script includes a sanity check feature that visualizes the conversion process, ensuring that the mask-to-YOLO conversion maintains shape integrity.

## Step 2: Setup Dataset Structure

Run the data setup script to organize the dataset for YOLOv8 training:

```bash
python data_setup.py
```

### Configuration Options:

Edit the `Config` class in `data_setup.py` to modify:

```python
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
        self.visualize_samples = 0  # Number of samples to visualize
        self.copy_strategy = 'copy'  # 'copy' or 'symlink'
        self.mode = "Depth"  # Options: None or "Depth"
```

### Depth Mode

The script supports special handling for depth maps with the `mode = "Depth"` option. When enabled, it:
- Automatically renames images to remove the `_depth` suffix
- Creates a proper directory structure compatible with YOLOv8
- Creates a temporary directory for processing
- Cleans up temporary files after completion

## Step 3: Train YOLOv8 Model

Run the main script to train the YOLOv8 model:

```bash
python main.py
```

### Configuration Options:

Edit the `Config` class in `main.py` to modify:

```python
class Config:
    def __init__(self):
        # Model settings
        self.model_name = 'yolov8s-seg.pt'  # Pre-trained YOLOv8s segmentation model
        
        # Paths
        self.data_yaml = '/kaggle/working/dataset.yaml'  # Path to dataset config
        self.output_path = '/kaggle/working/'
        self.project = '/kaggle/working/runs/segment'
        self.name = 'fragment_seg'
        
        # Training parameters
        self.epochs = 10
        self.imgsz = 512  # Image size: 512x512
        self.batch_size = 16
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # Class information
        self.classes = ['fragment']  # Single class: fragment
```

## Complete Workflow

For a complete workflow, run the scripts in sequence:

```bash
# 1. Convert masks to YOLO format
python mask_converter.py

# 2. Setup dataset structure
python data_setup.py

# 3. Train YOLOv8 model
python main.py
```

## Output and Model Weights

The training script will produce:
- Training metrics plot at `/kaggle/working/training_metrics.png`
- Best model weights at `/kaggle/working/best.pt`
- Full training logs and checkpoints in `/kaggle/working/runs/segment/fragment_seg`

### Model Weight Locations

The trained model weights are saved in two locations:
1. **Original location**: `/kaggle/working/runs/segment/fragment_seg/weights/best.pt` - Contains the best model based on validation metrics
2. **Copied location**: `/kaggle/working/best.pt` - A copy of the best model for easier access

Additionally, the last model checkpoint from the final epoch is saved at:
- `/kaggle/working/runs/segment/fragment_seg/weights/last.pt`

## Evaluation and Visualization

The training script automatically:
- Evaluates the model on the validation set
- Reports mAP50 and mAP50-95 metrics
- Visualizes predictions on a sample validation image
- Plots training metrics including mAP, training loss, and validation loss

## Using the Trained Model

After training, you can use the model for inference:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('/kaggle/working/best.pt')

# Run inference on an image
results = model('/path/to/test/image.jpg')

# Display the results
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.show()
```

## Transfer Learning

The training process uses transfer learning from a pre-trained YOLOv8s segmentation model. This approach:
- Leverages knowledge from large datasets
- Reduces training time
- Improves performance on small datasets

You can choose different pre-trained models by modifying the `model_name` parameter in the Config class:
- `yolov8n-seg.pt`: Nano model (smallest, fastest)
- `yolov8s-seg.pt`: Small model (balance of speed and accuracy)
- `yolov8m-seg.pt`: Medium model (more accurate)
- `yolov8l-seg.pt`: Large model (very accurate)
- `yolov8x-seg.pt`: XLarge model (most accurate)

## Customizing for Different Datasets

To use these scripts with a different dataset:

1. Modify the input paths in the `Config` classes to point to your dataset
2. Adjust the class names and number of classes if needed
3. Tune the training parameters based on your dataset size and complexity
