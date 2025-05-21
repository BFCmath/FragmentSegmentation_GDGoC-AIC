# ENEOPI Fragment Segmentation Server

## Server Folder Structure

Here's an overview of the key files and directories within the `server` folder:

### Backend
- `app.py`: The main FastAPI application file. It defines API endpoints, handles request processing, and integrates the model handlers.
- `model.py`: Contains the logic for loading models (ONNX format), preprocessing images, running inference, and post-processing the results to extract masks and calculate volumes.
- `config.py`: Stores configuration settings for the server and models, such as model paths, image dimensions, and logging preferences.
- `utils/depth_handler.py`: A utility module for handling depth map generation, using the Depth Anything V2 model.
- `utils/depth_anything_v2/`: Depth Anything V2 model files.
- `weights/`: This directory is crucial. It's where the pre-trained model weights must be placed.
- `requirements.txt`: Lists all Python dependencies required to run the server.

### Frontend
- `index.html`: The main HTML file for the frontend.
- `assets/main.js`: The main JavaScript file for the frontend.
- `assets/styles.css`: The CSS file for the frontend.

### Docker
- `Dockerfile`: Defines the instructions to build a Docker image for the server, encapsulating all dependencies and code.
- `docker-compose.yml`: A Docker Compose file for easily building and running the server container with predefined configurations, including volume mapping for weights.
- `.dockerignore`: Specifies files and directories to ignore when building the Docker image.

## Setup

### 1. Model Weights

The server requires pre-trained model weights to function.

**Download the weights:**

Please download the necessary model files from the following link:
[EnEoPi Weights](https://drive.google.com/drive/folders/1REDe3jkkYV856jkzJiQ5LFVdSxQ5qXeQ?usp=sharing)

**Place the weights:**

1. Create a directory named `weights` inside the `server` directory if it doesn't already exist:

    ```bash
    mkdir -p server/weights
    ```

2. Place the downloaded files directly into the `server/weights/` directory. The required files are:
    - `yolo_rgb_nano.onnx`
    - `yolo_rgbd_nano.onnx`
    - `depth_anything_v2_vits.pth`

### 2. Python Environment (for running without Docker)

If you choose not to use Docker, you'll need Python 3.9 or newer. It's highly recommended to use a virtual environment.

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

There are two main ways to run the server:

### Method 1: Using Docker (Recommended)

This is the easiest and recommended way to run the server as it handles all dependencies and environment setup automatically.

1. **Prerequisites:**
    - Docker installed.
    - Model weights placed in `server/weights/` as described above OR ensure the `docker-compose.yml` correctly volumes your external weights path. The provided `docker-compose.yml` mounts `./weights:/app/weights`, so ensure your weights are in a folder named `weights` in this directory.

2. **Build and Start:**
    Navigate to the `server` directory (where the `docker-compose.yml` file is located) and run:

    ```bash
    docker-compose up -d --build
    ```

    The `--build` flag ensures the image is rebuilt if there are changes. `-d` runs it in detached mode.

3. **Access the Server:**
   Open [http://localhost:3000](http://localhost:3000) in your web browser.

### Method 2: Running Directly with Python (Without Docker)

1. **Prerequisites:**
    - Python 3.9+ installed.
    - All dependencies installed from `requirements.txt` (see "Python Environment" section above).
    - Model weights placed in `server/weights/` as described in the "Model Weights Setup" section.

2. **Start the Server:**
    Navigate to the `server` directory and run:

    ```bash
    python app.py
    ```

    Alternatively, for development with auto-reload (if uvicorn is installed as per requirements):

    ```bash
    uvicorn app:app --host 127.0.0.1 --port 3000 --reload
    ```

    For production-like execution without Docker, matching the Docker CMD:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 3000
    ```

## Accessing the Server

Once the server is running (either via Docker or directly), you can access it in your web browser or through API clients.

**IMPORTANT NOTE:** Access the server API and UI via:
`http://localhost:3000`

Do not use `http://0.0.0.0:3000` in your browser.

## API Endpoints

- `GET /`: Serves the HTML homepage with a UI for interacting with the segmentation models.
- `POST /predict`: Endpoint for submitting images for segmentation.
  - **Request Body**: `multipart/form-data`
    - `file`: The image file (JPEG/PNG).
    - `use_depth` (query parameter, optional):
      - `"fast"` (default): Uses the RGB model.
      - `"precise"`: Uses the RGBD model for depth-enhanced segmentation.
  - **Response**: A JSON object containing:
    - `success` (boolean): True if processing was successful.
    - `image_data` (string, optional): Base64 encoded string of the processed image with segmentation masks.
    - `volumes` (list, optional): A list of calculated volumes for each detected segment.
    - `process_time_ms` (number, optional): Time taken for processing in milliseconds.
    - `error` (string, optional): Error message if processing failed.

## Environment Variables

The server can be configured using environment variables (especially relevant for Docker deployments):

- `PORT`: The port on which the server will listen (default: `3000`).
- `DEV`: Set to `true` or any value to enable development mode (e.g., more verbose logging, potential auto-reload features if configured in `app.py`). Default is `false`.
- `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`, `ERROR`). Default: `INFO`.
