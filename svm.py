# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Define function to load and preprocess the dataset
def load_dataset(data_dir):
    images = []
    labels = []

    for label in ['cat', 'dog']:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            image_path = os.path.join(label_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (100, 100))  # Resize images to a fixed size
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load dataset
data_dir = '/content/drive/MyDrive/path_to_your_dataset'
images, labels = load_dataset(data_dir)

# Flatten images
images_flat = images.reshape(images.shape[0], -1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print classification report
print(classification_report(y_test, y_pred))
