
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Robust imports: prefer tensorflow.keras
try:
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
except Exception:
    from keras.models import load_model, Model
    from keras.applications.resnet50 import preprocess_input
    from keras.preprocessing import image

import joblib
from PIL import Image
import io

# --- 1. SETUP: MODEL PATHS AND LABELS ---

MODEL_DIR = './models'
RESNET_MODEL_PATH = f'{MODEL_DIR}/fine_tuned_resnet50_SVM.h5'
SCALER_PATH = f'{MODEL_DIR}/scaler_SVM.pkl'
PCA_PATH = f'{MODEL_DIR}/pca_SVM.pkl'
SVM_PATH = f'{MODEL_DIR}/svm_face_shape2.pkl'

CLASS_LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
TARGET_SIZE = (224, 224)

# --- 2. Initialize Flask App and Load Models ---
app = Flask(__name__)
CORS(app)

resnet_model = None
feature_extractor = None
scaler = None
pca = None
svm_classifier = None

try:
    # Load fine-tuned ResNet50 model
    resnet_model = load_model(RESNET_MODEL_PATH, compile=False)
    print(f" ResNet50 model loaded from {RESNET_MODEL_PATH}")

    # Create feature extractor model (output from 'conv5_block3_out' layer)
    feature_extractor = Model(
        inputs=resnet_model.input,
        outputs=resnet_model.get_layer("conv5_block3_out").output
    )
    print(" Feature extractor model ready (from layer 'conv5_block3_out')")

    # Load Scaler, PCA, and SVM classifier
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    svm_classifier = joblib.load(SVM_PATH)

    print(f" Scaler, PCA, and SVM loaded successfully from {MODEL_DIR}")

except Exception as e:
    print(f" ERROR: Failed to load one or more models.")
    print(f"Error details: {e}")

# --- 3. Hairstyle Recommendation Function ---
def recommend_hairstyle(face_shape):
    recommendations = {
        "Heart": [
            "Side-swept bangs",
            "Chin-length bobs",
            "Soft layers around the chin"
        ],
        "Oblong": [
            "Wavy styles with volume",
            "Curtain bangs",
            "Shoulder-length cuts"
        ],
        "Oval": [
            "Almost any style suits",
            "Long waves",
            "Pixie cuts"
        ],
        "Round": [
            "Long layers to elongate the face",
            "Side parts",
            "High ponytails"
        ],
        "Square": [
            "Soft curls or waves",
            "Layered cuts that soften the jawline",
            "Side-swept fringes"
        ]
    }
    return recommendations.get(face_shape, ["Try consulting a stylist!"])

# --- 4. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image, extracts features using ResNet50, reduces with PCA, and classifies with SVM."""
    if any(x is None for x in [feature_extractor, scaler, pca, svm_classifier]):
        return jsonify({"error": "ML models failed to load on server start. Check console for details."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize(TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # --- Feature extraction ---
        features = feature_extractor.predict(x, verbose=0)
        features = features.reshape(features.shape[0], -1)

        # --- Scale and PCA transform ---
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # --- Predict using SVM ---
        probabilities = svm_classifier.predict_proba(features_pca)
        predicted_index = np.argmax(probabilities[0])
        predicted_shape = CLASS_LABELS[predicted_index]
        confidence = probabilities[0][predicted_index]

        # --- Hairstyle Recommendation ---
        hairstyles = recommend_hairstyle(predicted_shape)

        # Return JSON result
        return jsonify({
            "predicted_shape": predicted_shape,
            "confidence": f"{confidence:.3f}",
            "recommended_hairstyles": hairstyles
        })

    except Exception as e:
        print(f"Prediction failed due to error: {e}")
        return jsonify({"error": f"Internal prediction error: {e}"}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    print(" Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
