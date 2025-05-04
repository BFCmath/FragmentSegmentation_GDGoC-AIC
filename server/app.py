from contextlib import asynccontextmanager
from io import BytesIO
from os import environ as env
import time
from fastapi import FastAPI, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn


def homepage() -> HTMLResponse:
    with open("index.html", "r") as f:
        html_file = f.read()
        return HTMLResponse(html_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Defer model loading for faster startup
    from model import ModelHandler

    WEIGHT_PATH = "../weights/mask_rcnn_weight_0.pth"
    # Load the ML model
    app.state.model_handler = ModelHandler(WEIGHT_PATH)
    print(f"Model initialized successfully from {WEIGHT_PATH}")
    yield


async def predict(req: Request, file: UploadFile):
    """Process an image and return segmentation predictions"""

    model_handler = req.app.state.model_handler
        
    # Check if model is loaded
    if not model_handler:
        return Response(
            status_code=503,
            content="Model not initialized. Please try again later",
        )
        
    start_time = time.time()
        
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            return Response(
                status_code=403,
                content=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
            )
        
        # Read the image file
        contents = await file.read()
        if not contents:
            return Response(
                status_code=400,
                content=f"Empty file received"
            )
        
        # Process image
        predictions = model_handler.predict(contents)
        
        # Convert predictions to desired output format
        result = model_handler.postprocess(predictions)
        img = Image.fromarray(result)
        
        # Log processing time
        process_time = time.time() - start_time

        print(f"Processed {file.filename} in {process_time:.2f} seconds")

        img_out = BytesIO()
        img.save(img_out, format="PNG")
        img_out.seek(0)

        return StreamingResponse(img_out, media_type="image/png")
        
    except Exception as e:
        return Response(
            status_code=500,
            content=f"Error processing image: {e}"
        )


def make_app(is_dev: bool) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],    # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],    # Allows all methods
        allow_headers=["*"],    # Allows all headers
    )

    if is_dev:
        app.get("/")(homepage)
    else:
        # Cached response to avoid file i/o for every request
        res = homepage()
        app.get("/")(lambda: res)

    app.post("/predict")(predict)

    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    return app


# TODO: Extract to configuration file
dev = env.get("DEV") is not None
port = int(env.get("PORT", default="3000"))

app = make_app(dev)

if __name__ == "__main__":
    if dev:
        uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
