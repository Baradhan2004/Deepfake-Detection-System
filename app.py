from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. Flask Setup
# ==========================================
app = Flask(__name__)
CORS(app)  # allow frontend requests

# ==========================================
# 2. Device
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. Load Model (same as training)
# ==========================================
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("best_multigan_fast.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["REAL", "FAKE"]

# ==========================================
# 4. Transform (same as training)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# 5. Prediction Route
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']

    try:
        image_pil = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        real_prob = probs[0].item() * 100
        fake_prob = probs[1].item() * 100

        result = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = max(real_prob, fake_prob)

        # ==== GRAD-CAM ====
        # 1. Target layers
        target_layers = [model.features[-1]]
        
        # 2. Init CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # 3. Target class (0 for REAL, 1 for FAKE)
        target_class = 1 if result == "FAKE" else 0
        targets = [ClassifierOutputTarget(target_class)]
        
        # 4. Generate CAM
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        
        # 5. Overlay on original image
        img_rgb = np.array(image_pil.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        
        # 6. Encode to Base64
        cam_pil = Image.fromarray(cam_image)
        buffered = io.BytesIO()
        cam_pil.save(buffered, format="JPEG")
        cam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2),
            "real_prob": round(real_prob, 2),
            "fake_prob": round(fake_prob, 2),
            "heatmap": f"data:image/jpeg;base64,{cam_b64}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==========================================
# 6. Run Server
# ==========================================
if __name__ == '__main__':
    app.run(debug=True)
