from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging

from utils import classify_image  # Import from utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up logging
logging.basicConfig(level=logging.INFO)

class ImageData(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/home.html") as f:
        return f.read()

@app.post("/api/translate")
async def translate(image_data: ImageData):
    try:
        print("here")

        encoded = image_data.image
        logging.info(f"Received encoded image data length: {len(encoded)}")

        # Decode base64 to an image
        try:
            image = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
        except Exception as decode_error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        # Check if image decoding was successful
        if image is None:
            raise HTTPException(status_code=400, detail="Image decoding failed")

        # Classify the image using the function from utils.py
        predicted_class = classify_image(image)
        
        return JSONResponse(content={"predicted_class": predicted_class, "confidence": 100})
    
    except Exception as e:
        # Log the error for debugging purposes
        logging.error(f"Error in /api/translate: {str(e)}")
        raise HTTPException(status_code=400, detail="An error occurred during translation")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
