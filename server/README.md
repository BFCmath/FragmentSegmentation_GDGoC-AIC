# Fragment Segmentation API

This API provides image segmentation for fragment detection using a pre-trained Mask R-CNN model, with a user-friendly web interface for visualization.

## Overview

The system consists of:
- Backend API server running on Python with FastAPI
- Frontend web interface for image upload and visualization
- Fragment segmentation using a pre-trained Mask R-CNN model
- Visual representation of fragment size distribution with CDF plot

## Setup and Installation

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or use the provided Nix flake:

   ```bash
   nix develop
   ```

   Dependencies include:
   - uvicorn
   - fastapi
   - python-multipart
   - torch
   - torchvision

2. Make sure the model weight file is available at:

   ```bash
   weights/mask_rcnn_weight_0.pth
   ```

## Running the API

Start the API server with:

```bash
python app.py
```

The server will be available at <http://localhost:8000>

## Frontend Interface

Open `index.html` in a browser to access the web interface, which provides:

- Drag-and-drop or clipboard paste image upload
- Visualization of detected fragments with color highlighting
- Cumulative distribution function (CDF) plot of fragment sizes
- Interactive sizing display

## API Endpoints

### Health Check

- **GET /health**

Check if the API is running and if the model is loaded.

### Predict

- **POST /predict**

Upload an image for segmentation.

- Request: Form data with 'file' field containing an image (JPEG or PNG)
- Response: Image with segmentation masks that can be displayed directly in the frontend
- The segmentation identifies individual fragments and assigns them unique colors

## Example Usage

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("your_image.jpg", "rb")}
response = requests.post(url, files=files)
# For JSON response
predictions = response.json()
# For image response
with open("result.png", "wb") as f:
    f.write(response.content)
```

### Using the Web Interface

1. Open the index.html file in your browser
2. Click on the canvas or paste an image from clipboard
3. The image will be processed and displayed with detected fragments highlighted
4. The fragment size distribution will be shown as a CDF plot
