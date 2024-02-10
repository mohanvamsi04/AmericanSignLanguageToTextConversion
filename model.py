import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Define the path to your dataset
dataset_path = "D:\Project\Sign-Language-To-Text-and-Speech-Conversion\AtoZ_3.1"

# Load and preprocess the dataset
def load_and_preprocess_data(dataset_path):
    data = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))  # Resize images to a common size
            data.append(image)
            labels.append(label)
    data = np.array(data)
    data = data / 255.0  # Normalize pixel values to the range [0, 1]
    labels = to_categorical(labels)
    return data, labels

data, labels = load_and_preprocess_data(dataset_path)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model to a file
model.save("cnn8grps_rad1_model.h5")