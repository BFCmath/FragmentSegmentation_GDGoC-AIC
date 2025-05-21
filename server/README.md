# ENEOPI Blast Segmentation Server

API server for image segmentation using YOLO models with both RGB and RGBD (RGB + Depth) capabilities.

## Running with Docker

### Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)
- Model weights files in the expected format

### Model Weights Setup

Before building the Docker image, you need to ensure the model weight files are available. You have two options:

1. **Copy weights into the server/weights directory before building**:
   ```bash
   mkdir -p server/weights
   cp /path/to/your/weights/yolo_rgb_nano.pt server/weights/
   cp /path/to/your/weights/yolo_rgbd_nano.pt server/weights/
   cp /path/to/your/weights/depth_anything_v2_vits.pth server/weights/
   ```

2. **Mount the weights as a volume** when running the container (using docker-compose.yml):
   ```yaml
   volumes:
     - /path/to/your/weights:/app/weights
   ```

### Building and Running with Docker Compose (Recommended)

1. Navigate to the server directory:
   ```bash
   cd server
   ```

2. Build and start the server:
   ```bash
   docker-compose up -d
   ```

3. The server will be available at: http://localhost:3000

### Building and Running with Docker CLI

1. Build the Docker image:
   ```bash
   cd server
   docker build -t aic-gdsc-server .
   ```

2. Run the container:
   ```bash
   docker run -p 3000:3000 aic-gdsc-server
   ```

   Or with volume mounting for the weights:
   ```bash
   docker run -p 3000:3000 -v /path/to/your/weights:/app/weights aic-gdsc-server
   ```

## Environment Variables

The Docker container supports these environment variables:

- `PORT`: Server port (default: 3000)
- `DEV`: Development mode (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_TO_STDOUT`: Whether to log to stdout instead of file (default: true in Docker)

## API Endpoints

- `GET /`: Homepage with UI
- `POST /predict`: Process image with parameters:
  - `file`: Image file (JPEG/PNG)
  - `use_depth`: "fast" (RGB) or "precise" (RGBD) 