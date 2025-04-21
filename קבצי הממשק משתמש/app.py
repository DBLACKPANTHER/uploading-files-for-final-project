from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image, ImageFilter
from io import BytesIO
import os
import traceback
import gdown
from retinaface import RetinaFace

# === Auto-download model if not already present ===
model_file = "trained_age_model_v21.keras"
google_drive_id = "1UdxjwgVp0OXB8gFjt49njv24PmCz1xLD"

if not os.path.exists(model_file):
    print("\U0001F4E5 Downloading model from Google Drive...")
    gdown.download(id=google_drive_id, output=model_file, quiet=False)

print("\U0001F4E6 Loading model...")
model = tf.keras.models.load_model(model_file)
print("✅ Model loaded!")

# === Configuration ===
IMG_SIZE = (224, 224)
AGE_GROUPS = {
    0: "0-18",
    1: "19-40",
    2: "41-60",
    3: "61-80",
    4: "81+"
}

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        decoded = base64.b64decode(image_data)
        img = Image.open(BytesIO(decoded)).convert('RGB')
        original_img = img.copy()
        img_np = np.array(img)

        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(img_np)
        if not faces:
            return jsonify({"faces": [], "blurred_image": data['image']}), 200

        results = []

        for face_data in faces.values():
            x1, y1, x2, y2 = face_data["facial_area"]

            face_crop = img.crop((x1, y1, x2, y2))
            face_resized = face_crop.resize(IMG_SIZE)
            input_tensor = np.expand_dims(np.array(face_resized) / 255.0, axis=0)

            pred = model.predict(input_tensor, verbose=0)
            pred_class = int(np.argmax(pred))
            age_group = AGE_GROUPS.get(pred_class, "Unknown")

            if pred_class == 0:  # If under 18
                blurred = face_crop.filter(ImageFilter.GaussianBlur(30))
                original_img.paste(blurred, (x1, y1))

            results.append({
                "predicted_age_group": age_group,
                "coordinates": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            })

        # Convert modified image back to base64
        buffered = BytesIO()
        original_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_uri = f"data:image/jpeg;base64,{img_base64}"

        return jsonify({
            "faces": results,
            "blurred_image": image_uri
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Run the app on a specific host and port ===
if __name__ == '__main__':
    print("\U0001F680 Starting y_age_detection_web at http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=True, use_reloader=False)
