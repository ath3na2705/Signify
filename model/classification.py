import tensorflow as tf
import sys

sys.path.append(r"c:\users\a0105\appdata\local\programs\python\python313\lib\site-packages")
import cv2 # type: ignore
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('signify_ASL_image_classification_model_ver4.keras')

# Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

IMG_SIZE = 180  # Change this to match the training image size

while True:
    ret, frame = cap.read()  # Capture frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame (resize and normalize)
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)    # Add batch dimension

    # Debug: Display the frame to make sure it's being processed correctly
    cv2.imshow('Webcam Frame', frame)

    # Use the model to predict
    predictions = model.predict(img)
    print(f'Raw predictions: {predictions}')  # Check raw prediction output

    predicted_class = np.argmax(predictions)
    print(f'Predicted class: {predicted_class}')  # Check which class is predicted

    # Display the result on the frame
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('Webcam with Prediction', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()