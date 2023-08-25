import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# Load the trained model
model = models.load_model('trained_model.keras') # trained model name 

# Load the label-to-name mapping
label_to_name = {0: 'name1', 1: 'name2', 2: 'name3'}  # Update with your label-to-name mapping,can be made dynamic with detected labelling

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the detected face from the frame
        gray_face = gray_frame[y:y+h, x:x+w]

        # Preprocess the face image
        resized_face = cv2.resize(gray_face, (64, 64)) / 255.0
        input_face = np.expand_dims(resized_face, axis=-1)
        input_face = np.expand_dims(input_face, axis=0)

        # Make prediction using the model
        predictions = model.predict(input_face)
        predicted_label = np.argmax(predictions)
        predicted_name = label_to_name[predicted_label]

        # Draw the bounding box and name on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
