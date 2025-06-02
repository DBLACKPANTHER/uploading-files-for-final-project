from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from retinaface import RetinaFace
from datetime import datetime

# === 0Load model ===
MODEL_PATH = "age_model_vgg16_20250508_1223.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
print(f" Model loaded successfully: {os.path.abspath(MODEL_PATH)}")
000000
# === Flask setup ===
app = Flask(__name__, static_folder="static")
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AGE_GROUPS = {
    0: "0-18",
    1: "19-30",
    2: "31-50",
    3: "51-70",
    4: "71+"
}

# === Age prediction logic ===
def predict_age_class(image_np):
    img = cv2.resize(image_np, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    class_id = int(np.argmax(preds))
    return class_id

# === Serve HTML ===
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')
@app.route('/logo.png')
def logo():
    return send_from_directory('static', 'logo.png')
@app.route('/Header.png')
def Header():
    return send_from_directory('static', 'Header.png')

# === Predict endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        base64_str = data['image'].split(',')[1]
        decoded = base64.b64decode(base64_str)
        image = Image.open(BytesIO(decoded)).convert('RGB')
        img_rgb = np.array(image)

        print(f" Image received, shape: {img_rgb.shape}")

        # Save original image for debugging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"debug_input_{ts}.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Detect faces
        faces_info = RetinaFace.detect_faces(img_rgb)
        print("RetinaFace output:", faces_info)

        if not isinstance(faces_info, dict) or not faces_info:
            return jsonify({'error': 'No faces detected'}), 400

        result_img = img_rgb.copy()
        response = []

        for face_key, face_data in faces_info.items():
            area = face_data['facial_area']
            x1, y1, x2, y2 = area
            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                print(f"Skipping empty face area: {area}")
                continue

            class_id = predict_age_class(face_crop)
            age_group = AGE_GROUPS[class_id]
            print(f"Face {face_key}: Predicted age group = {age_group} (class {class_id})")

            response.append({
                "predicted_age_group": age_group,
                "coordinates": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

            if class_id == 0:
                blurred = cv2.GaussianBlur(face_crop, (99, 99), 30)
                result_img[y1:y2, x1:x2] = blurred
                print(f"\U0001F536 Blurred face at: {x1},{y1},{x2},{y2}")

        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        print("\u2705 Prediction complete.")
        return jsonify({
            "faces": response,
            "blurred_image": "data:image/jpeg;base64," + img_base64
        })

    except Exception as e:
        print(" Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# === Start server ===
if __name__ == '__main__':
    print(" Starting Flask API for Age Blur...")
    app.run(debug=True, port=7860)
