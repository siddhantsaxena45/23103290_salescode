import os
import io
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__)
device = torch.device("cpu")

# Load model globally to ensure low latency
print("Loading model...")
model = models.mobilenet_v3_small(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 1)

# Assumes best_model.pth is in the same directory
try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("Model loaded and ready!")
except Exception as e:
    print(f"Error loading model: {e}\nPlease ensure best_model.pth is in the same folder.")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    # Decode base64 image coming from the webcam
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    
    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    end = time.time()
    
    latency_ms = (end - start) * 1000
    
    # 0 = Real, 1 = Fake (Screen)
    result = "SCREEN (FAKE)" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    return jsonify({
        'prediction': result,
        'probability': prob,
        'confidence': f"{confidence*100:.2f}%",
        'latency_ms': f"{latency_ms:.2f}"
    })

if __name__ == '__main__':
    print("\n[+] Starting live demo server! Go to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
