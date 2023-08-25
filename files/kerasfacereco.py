import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load and preprocess the dataset
def load_dataset(data_dir):
    images = []
    labels = []
    
    label_to_name = {}  # Map labels to names
    
    # Loop through subdirectories (each subdirectory corresponds to a label)
    for label, person_name in enumerate(os.listdir(data_dir)):
        label_to_name[label] = person_name
        person_dir = os.path.join(data_dir, person_name)
        
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))  # Resize to a common size
            images.append(image)
            labels.append(label)
            print(image_path)
            print(label)
    return np.array(images), np.array(labels), label_to_name

data_dir = 'C:\\Users\\IT\\AppData\\Local\\Programs\\Python\\Python310\\venv4wxpython\\Scripts\\face recognition\\datasetnew'
images, labels, label_to_name = load_dataset(data_dir)

# Normalize images
images = images / 255.0

# Split dataset into training and testing sets
#split_ratio = 0.85
#split_index = int(len(images) * split_ratio)
#train_images, test_images = images[:split_index], images[split_index:]
#train_labels, test_labels = labels[:split_index], labels[split_index:]

# Split dataset into training, validation, and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_to_name), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
#history=model.fit(train_images, train_labels, epochs=30, validation_split=0.2)
# Train the model using the train_images and val_images for validation
history = model.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))

# Get training history from the history object
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
##till here potlib

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to names
predicted_names = [label_to_name[label] for label in predicted_labels]

# Display some predictions
for i in range(25):
    print(f"Predicted: {predicted_names[i]}, Actual: {label_to_name[test_labels[i]]}")



model.save('trained_model.keras')
