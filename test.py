from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('signify_ASL_image_classification_model.h5')

# Process the video frames and make predictions
def process_frame(frame):
    IMG_SIZE = 180  # Change to match your model's input size
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Endpoint to handle video input
@app.route('/process-video', methods=['POST'])
def process_video():
    file = request.files['video']
    npimg = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Process the frame and get the predicted class
    predicted_class = process_frame(frame)
    
    return jsonify({'predicted_class': str(predicted_class)})

# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
