from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import numpy as np
import cv2

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

class ImageData(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/home.html") as f:
        return f.read()

@app.post("/api/translate")
async def translate(image_data: ImageData):
    try:
        encoded = image_data.image
        image = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        predicted_class, confidence = classify_image(image)
        
        return JSONResponse(content={"predicted_class": int(predicted_class), "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
