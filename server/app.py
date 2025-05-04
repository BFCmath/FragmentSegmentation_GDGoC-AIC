from contextlib import asynccontextmanager
from io import BytesIO
import logging
from os import environ as env
import sys
import time

from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from model import ModelHandler


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

    # Load the ML model
    WEIGHT_PATH = '../weights/mask_rcnn_weight_0.pth'
    app.state.model_handler = ModelHandler(WEIGHT_PATH, app.state.logger)

    app.state.logger.info('Server successfully started')
    yield
    app.state.logger.info('Server shutting down...')


async def predict(req: Request, file: UploadFile):
    model_handler: ModelHandler = req.app.state.model_handler
    logger: logging.Logger = req.app.state.logger

    logger.info(f'Incoming request {req.client}')

    start_time = time.time()

    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png']:
            return Response(
                status_code=403,
                content=f'Invalid file type: {file.content_type}. Only JPEG and PNG are supported.'
            )
        
        # Read the image file
        contents = await file.read()
        if not contents:
            return Response(
                status_code=400,
                content=f'Empty file received'
            )
        
        # Process image
        predictions = model_handler.predict(contents)
        
        # Convert predictions to desired output format
        result = model_handler.postprocess(predictions)
        img = Image.fromarray(result)
        
        # Log processing time
        process_time = time.time() - start_time

        logger.info(f'Processed {req.client} in {process_time:.2f} seconds')

        img_out = BytesIO()
        img.save(img_out, format='PNG')
        img_out.seek(0)

        return StreamingResponse(img_out, media_type='image/png')
        
    except Exception as e:
        logger.error(f'Failed to process {req.client}: {e}')
        return Response(
            status_code=500,
            content=f'Error processing image: {e}'
        )


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
