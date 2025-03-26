import cv2
import numpy as np
import tensorflow as tf
from preprocess import preprocess_image

# Load trained model
MODEL_PATH = "../Models/fire_detector.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    print(f"Error loading model from {MODEL_PATH}. Train the model first.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = preprocess_image(frame)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]
    label = " Fire Detected!" if prediction > 0.5 else "No Fire"
    color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

    # Display result
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

