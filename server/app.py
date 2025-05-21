"""
FastAPI server for RGBD image analysis and segmentation.

This module implements a REST API for image segmentation using YOLO models with
both RGB and RGBD (RGB + Depth) capabilities. It handles image uploads, processes
them through the appropriate models, and returns annotated results.
"""

import base64
import logging
import sys
import time
from contextlib import asynccontextmanager
from io import BytesIO
from os import environ as env
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import ModelHandler, RGBDModelHandler
import config as cfg


# Configuration constants
# ALLOWED_CONTENT_TYPES = ['image/jpeg', 'image/png'] # Moved to config.py
# RGB_WEIGHT_PATH = '../weights/yolo_rgb_{}.pt' # Moved to config.py
# RGBD_WEIGHT_PATH = '../weights/yolo_rgbd_{}.pt' # Moved to config.py
# DEPTH_MODEL_PATH = '../weights/depth_anything_v2_vits.pth' # Moved to config.py


def homepage() -> HTMLResponse:
    """Return the HTML homepage for the application."""
    with open('index.html', 'r') as f:
        html_file = f.read()
        return HTMLResponse(html_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Handles initialization and cleanup of resources like logging and models.
    """
    # Configure logging
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format=cfg.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_FILE)
        ]
    )
    app.state.logger = logging.getLogger(__name__)

    # Load the models
    version_type = 'nano'  # Could be configurable via environment
    app.state.version_type = version_type
    
    try:
        # Initialize RGB model
        app.state.model_handler = ModelHandler(
            cfg.RGB_WEIGHT_PATH_TEMPLATE.format(version_type),
            app.state.logger
        )
        app.state.logger.info(f'Server successfully started with YOLO RGB {version_type} model')
        
        # Initialize RGBD model
        depth_model_filename = cfg.DEPTH_MODEL_PATH_TEMPLATE.format(cfg.DEPTH_MODEL_TYPE)
        app.state.rgbd_model_handler = RGBDModelHandler(
            cfg.RGBD_WEIGHT_PATH_TEMPLATE.format(version_type),
            depth_model_filename, # Use the constructed path
            app.state.logger
        )
        app.state.logger.info(f'RGBD model loaded successfully with depth model: {depth_model_filename}')
    except Exception as e:
        app.state.logger.error(f"Failed to initialize models: {e}")
        raise
        
    yield
    
    app.state.logger.info('Server shutting down...')


def _validate_file(file: UploadFile) -> Optional[JSONResponse]:
    """Validate uploaded file type and content."""
    if file.content_type not in cfg.ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "success": False,
                "error": f'Invalid file type: {file.content_type}. Only JPEG and PNG are supported.'
            }
        )
    return None

async def _read_file_contents(file: UploadFile) -> Optional[bytes]:
    """Read file contents. Returns None if file is empty."""
    contents = await file.read()
    if not contents:
        return None
    return contents

async def process_image(req: Request, file: UploadFile, use_depth: bool = False):
    """
    Process an uploaded image with either RGB or RGBD model.
    
    Args:
        req: FastAPI request object
        file: Uploaded image file
        use_depth: Whether to use depth estimation (RGBD model)
        
    Returns:
        JSONResponse with processed image data and volume measurements
    """
    logger: logging.Logger = req.app.state.logger

    model_handler = req.app.state.rgbd_model_handler if use_depth else req.app.state.model_handler
    model_type = "RGBD" if use_depth else f"YOLO {req.app.state.version_type}"
    logger.info(f'Incoming request {req.client} using {model_type} model')

    start_time = time.time()

    try:
        validation_response = _validate_file(file)
        if validation_response:
            return validation_response

        contents = await _read_file_contents(file)
        if contents is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Empty file received"}
            )
            
        predictions = model_handler.predict(contents)
        result, volumes = model_handler.postprocess(predictions)
        img = Image.fromarray(result)
        
        process_time = time.time() - start_time
        logger.info(f'Processed {req.client} with {model_type} model in {process_time:.2f} seconds')
        
        # Convert image to base64 for response
        img_out = BytesIO()
        img.save(img_out, format='PNG')
        img_out.seek(0)
        img_base64 = base64.b64encode(img_out.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "image_data": f"data:image/png;base64,{img_base64}",
            "volumes": volumes,
            "process_time_ms": round(process_time * 1000)
        })
        
    except Exception as e:
        logger.error(f'Failed to process {req.client} with {model_type} model: {e}')
        if use_depth: # Log traceback for RGBD failures, as they can be more complex
            import traceback
            logger.error(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


async def predict(req: Request, file: UploadFile, use_depth: Literal["fast", "precise"] = Query("fast")):
    """
    Process images using either fast (RGB) or precise (RGBD) mode.
    
    Args:
        req: FastAPI request object
        file: Uploaded image file
        use_depth: Whether to use precise mode with depth estimation
        
    Returns:
        Processed image results
    """
    use_depth_mode = use_depth.lower() == "precise"
    return await process_image(req, file, use_depth=use_depth_mode)


def make_app(is_dev: bool = False) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        is_dev: Whether the application is running in development mode
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        lifespan=lifespan,
        title="ENEOPI Blast Segmentation API",
        description="API for image segmentation using YOLO with RGB and RGBD capabilities",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Optimize homepage delivery in production
    if is_dev:
        app.get('/')(homepage)
    else:
        # Serve homepage dynamically to avoid caching issues, ensuring fresh assets
        app.get('/')(homepage)

    app.post('/predict')(predict)
    app.mount('/assets', StaticFiles(directory='assets'), name='assets')
    
    return app


# App configuration from environment variables
dev = env.get('DEV') is not None or cfg.DEV_MODE
port = int(env.get('PORT', default=str(cfg.PORT)))

app = make_app(dev)

if __name__ == '__main__':
    if dev:
        uvicorn.run('app:app', host='127.0.0.1', port=port, reload=True)
    else:
        uvicorn.run(app, host='0.0.0.0', port=port)
