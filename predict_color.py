import cv2
import joblib

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def predict_color(image_path, knn_model_path):
    knn = joblib.load(knn_model_path)
    image = cv2.imread(image_path)
    hist = extract_color_histogram(image)
    color = knn.predict([hist])[0]
    return color

# Example usage
image_path = "path_to_detected_object_image.jpg"
knn_model_path = "color_knn_classifier.pkl"
color = predict_color(image_path, knn_model_path)
print(f"The predicted color is: {color}")
