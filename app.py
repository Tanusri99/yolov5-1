from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
from json import dumps, loads
import cv2
import joblib
import numpy as np

app = Flask(__name__)

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: dumps(x).encode('utf-8'))
consumer = KafkaConsumer('color_detection_results',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset='earliest',
                         enable_auto_commit=True,
                         group_id='color-group',
                         value_deserializer=lambda x: loads(x.decode('utf-8')))

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

knn = joblib.load('color_knn_classifier.pkl')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    bbox = request.json.get('bbox')  # bounding box should be in [x, y, w, h] format
    
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    hist = extract_color_histogram(cropped_image)
    color = knn.predict([hist])[0]
    
    result = {'bbox': bbox, 'color': color}
    producer.send('color_detection_results', value=result)
    
    return jsonify(result)

@app.route('/results', methods=['GET'])
def results():
    messages = []
    for message in consumer:
        messages.append(message.value)
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True)
