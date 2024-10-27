import tensorflow as tf
import numpy as np
import sys

sys.path.append(r"c:\users\a0105\appdata\local\programs\python\python313\lib\site-packages")
import cv2# type: ignore

# Load the saved ASL Keras model
model = tf.keras.models.load_model('signify_ASL_image_classification_model_ver4.keras')

# Function to preprocess input image for classification
def preprocess_image(image):
    # Resize the image to 180x180 pixels (the size used for the ASL dataset)
    image = cv2.resize(image, (180, 180))
    
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    
    # Reshape the image to match the model input shape (1, 180, 180, 3)
    image = image.reshape(1, 180, 180, 3)
    
    return image

# Function to classify the ASL gesture in the image
def classify_image(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    
    # Predict the class probabilities
    predictions = model.predict(processed_image)
    
    # Get the predicted class (the one with highest probability)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence

# Real-time classification using webcam
def real_time_classification():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Define region of interest (ROI) for ASL gesture recognition
        roi = frame[100:400, 100:400]
        
        # Draw rectangle around ROI
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        try:
            # Classify the ASL gesture in the ROI
            predicted_class, confidence = classify_image(roi)
            
            # Display the prediction on the frame
            cv2.putText(frame, f"ASL: {predicted_class} ({confidence:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, str(e), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Real-Time ASL Classification', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    real_time_classification()
