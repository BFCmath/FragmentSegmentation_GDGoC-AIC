# YOLOv8 Instance Segmentation for Fragment Detection

This repository contains scripts to train a YOLOv8 model for instance segmentation of fragments. The workflow is divided into three independent scripts:

1. `mask_converter.py` - Converts colorful mask images to YOLO segmentation format
2. `data_setup.py` - Sets up the dataset structure for YOLOv8 training
3. `main.py` - Trains a YOLOv8 segmentation model
4. `inference.py` - Performs inference on images using the trained YOLOv8 model

