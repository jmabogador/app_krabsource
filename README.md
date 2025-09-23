## Synopsis

***KrabSource*** is a mobile and web-based application that uses a Convolutional Neural Network (CNN), specifically the ResNet-18 model, to identify and classify crab species found in Panay Island, Philippines. The system classifies crabs based on patterns, colors, and shapes, integrates GPS for mapping, and supports real-time monitoring of species. Its goal is to automate crab identification, improve ecological research, and assist in marine biodiversity conservation.

## Code Example
```
from flask import Flask, request, jsonify
from fastai.vision.all import load_learner
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os

# Initialize Flask
app = Flask(__name__)

# Load trained ResNet-18 model
model_path = os.path.join(os.path.dirname(__file__), 'krabsource.pkl')
learn = load_learner(model_path)`

# Define color ranges (HSV) for crab detection
color_ranges = {
    'Blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
    'Green': (np.array([30, 50, 50]), np.array([80, 255, 255])),
    'Brown': (np.array([0, 50, 50]), np.array([20, 255, 150])),
    'Red': (np.array([0, 50, 50]), np.array([20, 255, 255])),
    'Yellow': (np.array([20, 50, 50]), np.array([40, 255, 255])),
    'Orange': (np.array([10, 100, 100]), np.array([20, 255, 255])),
}

def preprocess_image(image_file):
    """Resize image to 224x224 for ResNet-18 input"""
    img = Image.open(BytesIO(image_file.read()))
    return img.resize((224, 224))

def detect_color(image):
    """Find the dominant color in the crab image"""
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    color_areas = {c: np.sum(cv2.inRange(hsv, l, u))
                   for c, (l, u) in color_ranges.items()}
    return max(color_areas, key=color_areas.get, default="Unknown")

@app.route("/classify", methods=["POST"])
def classify_image():
    """Classify crab species and return details"""
    image_file = request.files["image"]
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    # Preprocess + predict
    img = preprocess_image(image_file)
    pred_class, pred_idx, outputs = learn.predict(img)
    confidence = float(outputs[pred_idx]) * 100`

    # Low confidence handling
    if confidence < 60:
        return jsonify({"Message": "Take a clearer image", "Confidence": confidence})

    # Extract additional info
    dominant_color = detect_color(img)

    return jsonify({
        "Species": str(pred_class),
        "Confidence": confidence,
        "Color": dominant_color,
        "Location": {"lat": latitude, "lon": longitude}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```
### Example Usage (cURL)
```
curl -X POST http://localhost:5000/classify \
  -F "image=@sample_crab.jpg" \
  -F "latitude=10.7393" \
  -F "longitude=122.5675"
```

### *Response:*
```
{
  "Species": "Kasag (male)",
  "Confidence": 97.85,
  "Color": "Blue",
  "Location": {"lat": "10.7393", "lon": "122.5675"}
}
```
## Motivation

Traditional crab identification is manual, time-consuming, and prone to human error. Skilled taxonomists are limited, making large-scale monitoring costly and inconsistent. By leveraging AI and mobile technology, KrabSource provides an accessible, accurate, and scalable way to classify and map crab species, empowering researchers, fishermen, NGOs, and government agencies to make informed decisions for marine conservation.

## Installation

- ### Download the APK
> Get the KrabSource mobile application APK from the official distribution source.
> You can download the app [`/dist`](./dist) directly in this folder.

- ### Enable Unknown Sources
> On your Android device, go to Settings > Security and enable Install from Unknown Sources.

- ### Install the Application
> Open the downloaded APK file and follow the on-screen prompts to install.

- ### Launch the App
> After installation, open KrabSource from your app drawer.

- ### Grant Permissions
> Make sure to allow access to your device’s camera, storage, and GPS location when prompted.

## API Reference

[Server Link](https://github.com/ThirdyNeko/Krabsource_flask.git)

## Tests

Create a file named test_app.py:
```
import io
import pytest
from app import app  # assuming your main Flask file is app.py

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_classify_endpoint(client):
    # Load a sample crab image for testing
    with open("sample_crab.jpg", "rb") as img_file:
        data = {
            "image": (io.BytesIO(img_file.read()), "sample_crab.jpg"),
            "latitude": "10.7393",
            "longitude": "122.5675"
        }
        response = client.post("/classify", data=data, content_type="multipart/form-data")

    # Verify response
    assert response.status_code == 200
    json_data = response.get_json()

    # Expected keys in response
    assert "Species" in json_data
    assert "Confidence" in json_data
    assert "Color" in json_data
    assert "Location" in json_data
    assert "lat" in json_data["Location"]
    assert "lon" in json_data["Location"]

    # Confidence should be between 0–100
    assert 0 <= json_data["Confidence"] <= 100
```
### Run Tests
```
pytest test_app.py -v
```
### Expected Output (example)
```
test_app.py::test_classify_endpoint PASSED
```
## Contributors

### Documentation
- Fork the repository.  
- Create a feature branch (`git checkout -b feature-name`).  
- Make your changes.  
- Commit and push your branch.  
- Submit a pull request for review.

### Stay Connected
- Message us on [Email](ireneo.catequista@wvsu.edu.ph).
