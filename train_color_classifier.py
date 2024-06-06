import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

def extract_color_histogram(image, bins=(8, 8, 8)):
    # Convert the image to the HSV color space and compute a 3D color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Paths to training dataset
data_dir = "/home/service/yolov5/training_dataset"
colors = ["black", "white", "red", "green", "blue", "yellow", "orange"]

# Initialize lists to hold features and labels
features = []
labels = []

# Loop over the color folders
for color in colors:
    color_dir = os.path.join(data_dir, color)
    for image_name in os.listdir(color_dir):
        image_path = os.path.join(color_dir, image_name)
        image = cv2.imread(image_path)
        hist = extract_color_histogram(image)
        features.append(hist)
        labels.append(color)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, labels)

# Save the model
joblib.dump(knn, 'color_knn_classifier.pkl')
print("Color classifier trained and saved as color_knn_classifier.pkl")
