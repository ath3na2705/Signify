from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import base64
import cv2

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ASL classification model
model = tf.keras.models.load_model("model/signify_ASL_image_classification_model_ver4.keras")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for the request body
class ImageData(BaseModel):
    image: str

# Root route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/home.html") as f:
        return f.read()

# Endpoint for translation
@app.post("/api/translate")
async def translate(image_data: ImageData):
    try:
        # Process the incoming base64 image
        encoded = image_data.image  # Get the base64 image string
        image = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image for the model
        image = cv2.resize(image, (224, 224))  # Adjust this to your model's input shape
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=-1)

        return JSONResponse(content={"predicted_class": predicted_class.tolist()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
