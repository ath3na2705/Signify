from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import base64
from pydantic import BaseModel
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('signify_ASL_image_classification_model_ver4.keras')

# Define a data model for the request payload
class ImageData(BaseModel):
    image: str  # base64-encoded image data

# Function to preprocess the image
def preprocess_image(image: np.ndarray):
    IMG_SIZE = 180  # Size the model expects
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype("float32") / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image: np.ndarray) -> int:
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions))
    return predicted_class

# Endpoint to receive base64-encoded image and return prediction
@app.post("/api/translate")
async def translate(image_data: ImageData):
    # Decode the base64 image data
    try:
        image_bytes = base64.b64decode(image_data.image)
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # Predict the class
        predicted_class = predict(image)
        
        # Return prediction as JSON
        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
