import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Danh sách lớp (phải đúng thứ tự lúc train)
CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper',
    'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear',
    'peas', 'pineapple', 'pomegranate', 'potato',
    'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# ==========================
# Load model
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "best_resnet50.h5")
)

print("Loading model from:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model tại:\n{MODEL_PATH}"
    )

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================
# Hàm dự đoán
# ==========================
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)

    return preds


# ==========================
# Trang chủ
# ==========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ==========================
# API dự đoán
# ==========================
@app.route("/predict", methods=["POST"])
def upload():

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]

    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f.filename)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    f.save(file_path)
    print(f"File saved to: {file_path}")

    try:
        print("Starting prediction...")
        preds = model_predict(file_path, model)
        print("Prediction finished.")

        pred_indices = np.argsort(preds[0])[::-1][:3]

        results = []

        for idx in pred_indices:
            results.append({
                "label": CLASSES[idx],
                "confidence": round(float(preds[0][idx]) * 100, 2)
            })

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "predictions": results
        })

    except Exception as e:

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "error": str(e)
        }), 500


# ==========================
# Run Flask
# ==========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )