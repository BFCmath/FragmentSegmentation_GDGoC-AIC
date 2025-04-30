# Fragment Segmentation API

This API provides image segmentation for fragment detection using a pre-trained Mask R-CNN model.

## Setup and Installation

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the model weight file is available at:

   ```bash
   weights/mask_rcnn_weight_0.pth
   ```

## Running the API

Start the API server with:

```bash
python app.py
```

The server will be available at <http://localhost:3000>

## API Endpoints

### Health Check

- **GET /health**

Check if the API is running and if the model is loaded.

### Predict

- **POST /predict**

Upload an image for segmentation.

- Request: Form data with 'file' field containing an image (JPEG or PNG)
- Response: JSON object containing:
  - boxes: List of bounding boxes
  - scores: List of confidence scores
  - labels: List of class labels
  - masks: List of segmentation masks
  - combined_mask: Combined segmentation mask

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
predictions = response.json()
```
