from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import base64
import cv2

# Initialize the FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ASL classification model
model = tf.keras.models.load_model("signify_ASL_image_classification_model_ver4.keras")

@app.post("/api/translate")
async def translate(image_data: str):
    try:
        # Decode the base64 image data
        header, encoded = image_data.split(",", 1)
        image = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Preprocess the image as needed for your model
        # For example, resizing and normalizing
        image = cv2.resize(image, (224, 224))  # Adjust size to your model's input shape
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=-1)

        return JSONResponse(content={"class": predicted_class.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
