# Model Training for ENEOPI Fragment Segmentation

This directory contains the Jupyter Notebooks and Python scripts used for training the segmentation and depth estimation models that power the ENEOPI Fragment Segmentation server.
The primary goal of this workflow is to produce the `.pt` model weights, which are subsequently converted to `.onnx` format for efficient inference by the server.

## Folder Structure Overview

- `fst-yolov8-rgb.ipynb`: Jupyter Notebook for training the YOLOv8-nano segmentation model on standard RGB images.
- `fst-depth-data-generation.ipynb`: Jupyter Notebook for generating depth maps from RGB images. These depth maps serve as the fourth channel for the RGB-D dataset.
- `fst-mask-converter.ipynb`: Jupyter Notebook to convert segmentation masks into the YOLO `.txt` format, required for training the YOLOv8 RGB-D model.
- `fst-yolov8-rgbd.ipynb`: Jupyter Notebook for training the 4-channel YOLOv8-nano segmentation model using RGB images combined with the generated depth data.
- `script/`: Contains various Python scripts that likely provide utility functions, helper classes, or modular components used within the training notebooks (e.g., dataset handling, model definitions, specific training loops).
- `depth_estimation/`: Contains scripts or utilities related to the Depth Anything V2 model.
- `version/`: Potentially holds results for different model versions.

## Model Training Workflow & Replication

To replicate the model training process and generate the weights, follow the steps outlined below by running the specified Jupyter Notebooks in sequence. It is recommended to use Kaggle.

### 1. Train YOLOv8 RGB Segmentation Model

- **Notebook:** `fst-yolov8-rgb.ipynb`
- **Purpose:** This notebook trains a YOLOv8-nano segmentation model using standard RGB images of blast fragments.
- **Process:** It loads the RGB image dataset, defines the model configuration, runs the training loop, and evaluates the model.
- **Output:** A `.pt` weight file (e.g., `yolo_rgb_nano.pt`) for the RGB segmentation model.

### 2. Generate Depth Data for RGB-D Model Input

- **Notebook:** `fst-depth-data-generation.ipynb`
- **Purpose:** This notebook takes your RGB image dataset and uses a pre-trained monocular depth estimation model (Depth Anything V2) to generate corresponding depth maps.
- **Process:** For each RGB image, it infers a depth map. This depth map will later be used as the fourth input channel for the RGB-D segmentation model.
- **Input:** RGB image dataset.
- **Output:** A dataset of depth maps, corresponding to the input RGB images.

### 3. Convert Masks to YOLO Format for RGB-D Training

- **Notebook:** `fst-mask-converter.ipynb`
- **Purpose:** This notebook converts your existing segmentation ground truth masks (e.g., from COCO JSON, PNG masks, etc.) into the `.txt` file format required by the YOLOv8 training API.
- **Process:** It reads the original mask annotations and outputs a `.txt` file for each image, where each line represents an object and contains the class ID followed by the normalized polygon coordinates of the mask.
- **Input:** Original segmentation masks and image dataset.
- **Output:** YOLO-compatible `.txt` annotation files for the RGB-D training dataset.

### 4. Train YOLOv8 RGB-D Segmentation Model

- **Notebook:** `fst-yolov8-rgbd.ipynb`
- **Purpose:** This notebook trains a 4-channel YOLOv8-nano segmentation model.
- **Process:** It uses the original RGB images and the depth maps generated in Step 2 as a 4-channel input. The ground truth annotations are the `.txt` files generated in Step 3. The model is trained to segment fragments using this combined RGB-D information.
- **Input:** RGB images, generated depth maps, and YOLO `.txt` annotations.
- **Output:** A `.pt` weight file (e.g., `yolo_rgbd_nano.pt`) for the RGB-D segmentation model.

### 5. Depth Estimation Model (Depth Anything V2)

- **Model:** Depth Anything V2 (specifically, the "Small" version)
- **Source:** The weights for this model are pre-trained and provided by the authors.
- **Action:** Download the `Depth-Anything-V2-Small` checkpoint (`depth_anything_v2_vits.pth`).
  - **Official GitHub Repository:** [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Usage:** These weights are used by the `fst-depth-data-generation.ipynb` notebook (Step 2) and by the `utils/depth_handler.py` in the server for on-the-fly depth map generation if RGBD inference is requested without pre-generated depth maps.

## Output

The primary outputs of this model training workflow are the `.pt` (PyTorch) model weight files:
- `yolo_rgb_nano.pt`
- `yolo_rgbd_nano.pt`

These `.pt` files are then typically converted to the `.onnx` format (e.g., `yolo_rgb_nano.onnx`, `yolo_rgbd_nano.onnx`) for use in the server application. The conversion step ensures optimized and portable models for inference.
