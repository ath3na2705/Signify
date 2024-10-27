import tensorflow as tf
import numpy as np
import cv2

# Load the ASL Keras model (ensure the model is only loaded once)
model = tf.keras.models.load_model('models/signify_ASL_image_classification_model_ver4.keras')

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (180, 180))  # Adjust to the model's input shape
    image = image / 255.0
    image = image.reshape(1, 180, 180, 3)
    return image

# Function to classify an ASL gesture in an image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return predicted_class, confidence