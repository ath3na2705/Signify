import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp


# Load the ASL Keras model (ensure the model is only loaded once)
model = tf.keras.models.load_model('models/signify_ASL_image_classification_model_ver4.h5')

# Function to preprocess the input image
# Load Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to classify an ASL gesture in an image
def classify_image(image):
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
          
        image = cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y, lm.z])

                # Flatten and reshape
                input_data = np.array(landmark_list).flatten().reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)
                predicted_label = chr(predicted_index + ord('A')) if predicted_index < 26 else '-'
                print("predicted = ",predicted_label)
                return predicted_label
    return '-'



