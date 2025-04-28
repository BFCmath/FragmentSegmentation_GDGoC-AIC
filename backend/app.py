import os
import io
import logging
import sys
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import traceback
import numpy as np
from model import ModelHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Fragment Segmentation API",
    description="API for instance segmentation of fragments using Mask R-CNN",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model handler
WEIGHT_PATH = "../weights/mask_rcnn_weight_0.pth"
model_handler = None

@app.on_event("startup")
async def startup_event():
    """Initialize model when application starts"""
    global model_handler
    try:
        # Create the ModelHandler instance
        model_handler = ModelHandler(WEIGHT_PATH)
        # Load the model
        model_handler.load()
        logger.info(f"Model initialized successfully from {WEIGHT_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())

@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running"""
    global model_handler
    if model_handler and model_handler.initialized:
        return {"status": "healthy", "model_loaded": True}
    elif model_handler:
        return {"status": "unhealthy", "error": "Model not loaded"}
    else:
        return {"status": "unhealthy", "error": "Model handler not initialized"}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Process an image and return segmentation predictions"""
    global model_handler
    
    # Check if model is loaded
    if not model_handler or not model_handler.initialized:
        raise HTTPException(
            status_code=503, 
            detail="Model not initialized. Please try again later."
        )
    
    # Log request info
    client_ip = request.client.host
    logger.info(f"Request received from {client_ip} for file {file.filename}")
    
    start_time = time.time()
    
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
            )
        
        # Read the image file
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        # Process image
        predictions = model_handler.predict(contents)
        
        # Convert predictions to desired output format
        result = model_handler.postprocess(predictions)
        
        # Log processing time
        process_time = time.time() - start_time
        logger.info(f"Processed {file.filename} in {process_time:.2f} seconds")
        
        # Return the response
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
