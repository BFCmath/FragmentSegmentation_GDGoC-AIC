# ENEOPI Blast Segmentation Server

This server provides a REST API for image segmentation using YOLO models, with both RGB and RGBD (RGB + Depth) capabilities. It handles image uploads, processes them through the appropriate models, and returns annotated results with volume measurements.

## Features

- Real-time image segmentation with YOLO models
- Depth estimation using Depth Anything V2
- Combined RGB+D processing for more accurate segmentation
- RESTful API for easy integration
- Interactive web interface for testing

## System Requirements

- Python 3.9+ recommended
- CUDA-compatible GPU recommended for faster processing
- 4GB+ RAM
- 1GB+ disk space for model weights

## Installation

1. Clone this repository and navigate to the server directory:
```bash
git clone https://github.com/your-organization/eneopi-blast-segmentation.git
cd eneopi-blast-segmentation/server
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Install the Depth Anything V2 model:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
mv Depth-Anything-V2 utils/depth_anything_v2
```

4. Download the model weights:
   - Create a `weights` directory in the parent folder
   - Download the following weights files:
     - YOLO RGB model: `yolo_rgb_nano.pt`
     - YOLO RGBD model: `yolo_rgbd_nano.pt`
     - Depth Anything V2 model: `depth_anything_v2_vits.pth`
   
   Place them in the `../weights/` directory.

## Running the Server

### Development Mode

```bash
# Enable development mode with auto-reload
export DEV=1
python app.py
```

### Production Mode

```bash
python app.py
# Or specify a custom port
PORT=5000 python app.py
```

## API Endpoints

### `POST /predict`

Processes an image and returns segmentation results.

#### Parameters

- `file`: The image file to process (JPEG or PNG)
- `use_depth`: Mode to use - "fast" (RGB only) or "precise" (RGBD)

#### Response

JSON object containing:
- `success`: Boolean indicating if processing was successful
- `image_data`: Base64-encoded processed image with segmentation
- `volumes`: Array of calculated volumes for each detected segment
- `process_time_ms`: Processing time in milliseconds

## Project Structure

```
server/
├── app.py                 # FastAPI application and route definitions
├── model.py               # Model handlers for RGB and RGBD inference
├── index.html             # Web interface
├── requirements.txt       # Python dependencies
├── assets/                # Static assets (JS, CSS)
└── utils/
    ├── depth_handler.py   # Depth estimation handler
    └── depth_anything_v2/ # Depth Anything V2 model
```

## Future Enhancements

- ONNX model export for faster inference
- Docker containerization
- Authentication system
- MinIO integration for object storage

## License

MIT License
