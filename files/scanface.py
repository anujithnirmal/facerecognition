import os
import cv2

# Create a directory to store the dataset
dataset_dir = 'C:\\Users\\IT\\AppData\\Local\\Programs\\Python\\Python310\\venv4wxpython\\Scripts\\face recognition\\datasetnew'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the default camera (0)

# Counter for image filenames
image_counter = 0

# Label for the person in the images
person_label = input("Enter person's name: ")

new_folder=person_label
new_path = os.path.join(dataset_dir, new_folder)
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Loop to capture images
while True:
    ret, frame = camera.read()

    # Display the current frame
    cv2.imshow('Capture Images', frame)

    # Press 's' to save the current frame as an image
    key = cv2.waitKey(1)
    if key == ord('s'):
        image_filename = os.path.join(new_path, f'{person_label}_{image_counter}.jpg')
        cv2.imwrite(image_filename, frame)
        print(f'Saved: {image_filename}')
        image_counter += 1
    elif key == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
