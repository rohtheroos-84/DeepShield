import os
import json
import tempfile

import cv2
import numpy as np
import torch
import timm
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Max upload size: 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the model architecture using timm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model = model.to(device)

# Function to remove "module." prefix from state_dict keys (from DataParallel training)
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

# Load the model weights using a path relative to this file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_vit_model.pth")

if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("Please place your trained model at: backend/models/best_vit_model.pth")
else:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {MODEL_PATH}")

model.eval()

# ImageNet normalization values (must match training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame):
    """Preprocess a single BGR frame for the ViT model."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 224x224
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    # Normalize to [0, 1] then apply ImageNet normalization
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_normalized = (frame_normalized - IMAGENET_MEAN) / IMAGENET_STD
    # Convert to tensor: (1, 3, 224, 224)
    tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    video_file = request.files["file"]

    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        video_file.save(temp.name)
        video_path = temp.name

    def generate():
        cap = cv2.VideoCapture(video_path)
        real_count = 0
        fake_count = 0
        total_frames = 0

        # Get total frame count and FPS for frame-skipping
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process ~2 frames per second to avoid being excessively slow
        skip = max(1, int(fps / 2))

        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                # Skip frames to speed up processing
                if frame_idx % skip != 0:
                    continue

                total_frames += 1
                input_tensor = preprocess_frame(frame)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    # Label mapping: 0 = Real, 1 = Manipulated/Fake
                    if predicted.item() == 0:
                        real_count += 1
                    else:
                        fake_count += 1

                data = {
                    "total_frames": total_frames,
                    "real_count": real_count,
                    "fake_count": fake_count,
                    "real_percentage": (real_count / total_frames) * 100,
                    "fake_percentage": (fake_count / total_frames) * 100,
                }
                yield f"data: {json.dumps(data)}\n\n"
        finally:
            cap.release()
            # Always clean up temp file
            try:
                os.remove(video_path)
            except OSError:
                pass

    return Response(generate(), mimetype="text/event-stream")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": os.path.exists(MODEL_PATH),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
