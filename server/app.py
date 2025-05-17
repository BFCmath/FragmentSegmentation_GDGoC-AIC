from contextlib import asynccontextmanager
from io import BytesIO
import logging
from os import environ as env
import sys
import time
import base64
from PIL import Image
from fastapi import FastAPI, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from model import RGBDModelHandler, ModelHandler


def homepage() -> HTMLResponse:
    with open('index.html', 'r') as f:
        html_file = f.read()
        return HTMLResponse(html_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('api.log')
        ]
    )
    app.state.logger = logging.getLogger(__name__)

    # Load the RGB YOLO model
    version_type = 'nano' 
    app.state.version_type = version_type
    
    RGB_WEIGHT_PATH = f'../weights/yolo_rgb_{version_type}.pt'
    RGBD_WEIGHT_PATH = f'../weights/yolo_rgbd_{version_type}.pt'
    DEPTH_MODEL_PATH = '../weights/depth_anything_v2_vits.pth'
    
    try:
        app.state.model_handler = ModelHandler(RGB_WEIGHT_PATH, app.state.logger)
        app.state.logger.info(f'Server successfully started with YOLO RGB {version_type} model')
        
        app.state.rgbd_model_handler = RGBDModelHandler(RGBD_WEIGHT_PATH, DEPTH_MODEL_PATH, app.state.logger)
        app.state.logger.info(f'RGBD model loaded successfully')
    except Exception as e:
        app.state.logger.error(f"Failed to initialize models: {e}")
        raise
        
    yield
    app.state.logger.info('Server shutting down...')


async def process_image(req: Request, file: UploadFile, use_depth: bool = False):
    """Process an image with either RGB or RGBD model based on parameter"""
    logger: logging.Logger = req.app.state.logger

    # Select the appropriate model handler
    model_handler = req.app.state.rgbd_model_handler if use_depth else req.app.state.model_handler
    model_type = "RGBD" if use_depth else f"YOLO {req.app.state.version_type}"
    logger.info(f'Incoming request {req.client} using {model_type} model')

    start_time = time.time()

    try:       
        if file.content_type not in ['image/jpeg', 'image/png']:
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error": f'Invalid file type: {file.content_type}. Only JPEG and PNG are supported.'
                }
            )
        
        contents = await file.read()
        if not contents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Empty file received"
                }            
            )
            
        predictions = model_handler.predict(contents)
        result, volumes = model_handler.postprocess(predictions)
        img = Image.fromarray(result)
        
        process_time = time.time() - start_time
        logger.info(f'Processed {req.client} with {model_type} model in {process_time:.2f} seconds')
        img_out = BytesIO()
        img.save(img_out, format='PNG')
        img_out.seek(0)
        
        img_base64 = base64.b64encode(img_out.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "image_data": f"data:image/png;base64,{img_base64}",
            "volumes": volumes
        })
        
    except Exception as e:
        logger.error(f'Failed to process {req.client} with {model_type} model: {e}')
        if use_depth:
            import traceback
            logger.error(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

async def predict(req: Request, file: UploadFile, use_depth: str = Query("fast")):
    use_depth_mode = use_depth.lower() == "precise"
    return await process_image(req, file, use_depth=use_depth_mode)

def make_app(is_dev: bool) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],    # Allows all origins
        allow_credentials=True,
        allow_methods=['*'],    # Allows all methods
        allow_headers=['*'],    # Allows all headers
    )

    if is_dev:
        app.get('/')(homepage)
    else:
        # Cached response to avoid file I/O for every request
        res = homepage()
        app.get('/')(lambda: res)

    app.post('/predict')(predict)

    app.mount('/assets', StaticFiles(directory='assets'), name='assets')
    return app


# TODO: Extract to configuration file
dev = env.get('DEV') is not None
port = int(env.get('PORT', default='3000'))

app = make_app(dev)

if __name__ == '__main__':
    if dev:
        uvicorn.run('app:app', host='127.0.0.1', port=port, reload=True)
    else:
        uvicorn.run(app, host='0.0.0.0', port=port)
