import tensorflow as tf
import numpy as np
import sys

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
        
        roi = frame[100:400, 100:400]  # Define ROI for ASL gesture recognition
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        try:
            predicted_class, confidence = classify_image(roi)
            cv2.putText(frame, f"ASL: {predicted_class} ({confidence:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, str(e), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Real-Time ASL Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_classification()