import sys
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")

# Load model
model = models.mobilenet_v3_small(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 1)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

if len(sys.argv) < 2:
    print("Usage: python predict.py image.jpg")
    sys.exit()

img_path = sys.argv[1]

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

start = time.time()

with torch.no_grad():
    output = model(img)
    prob = torch.sigmoid(output).item()

end = time.time()

latency_ms = (end - start) * 1000
print(prob)
