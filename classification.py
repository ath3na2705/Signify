import tensorflow as tf
import numpy as np
import sys

def predict_sign():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                    # Display the prediction
                    cv2.putText(image, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('ASL Prediction', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Run real-time prediction
predict_sign()