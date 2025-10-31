import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Robust imports: prefer tensorflow.keras, but fall back to standalone keras if needed
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing import image
except Exception:
    from keras.models import load_model
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image

import joblib 
from PIL import Image
import io

# --- 1. SETUP: MODEL PATHS AND LABELS ---

# IMPORTANT: These paths MUST match where you saved the files in Step 1.2
DT_MODEL_PATH = './saved_models/decision_tree_clf.pkl'
VGG_MODEL_PATH = './saved_models/vgg16_base_model.h5'

# Define your class labels in the correct order (based on your generator)
CLASS_LABELS = ['Square', 'Round', 'Oval', 'Heart', 'Oblong'] 
TARGET_SIZE = (224, 224) 

# --- 2. Initialize App and Load Models ---
app = Flask(__name__)
# Enable CORS to allow React (e.g., on port 3000) to talk to Flask (on port 5000)
CORS(app) 

vgg_base_model = None
dt_classifier = None

try:
    # Load the VGG16 base model structure and weights
    vgg_base_model = load_model(VGG_MODEL_PATH, compile=False) 
    print(f"✅ VGG16 Model loaded from {VGG_MODEL_PATH}")

    # Load the trained Decision Tree classifier
    dt_classifier = joblib.load(DT_MODEL_PATH)
    print(f"✅ Decision Tree Classifier loaded from {DT_MODEL_PATH}")

except Exception as e:
    print(f"❌ ERROR: Could not load models. Ensure files are saved correctly in the 'saved_models' directory.")
    print(f"Error details: {e}")


# --- 3. Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives image, extracts features using VGG16, and classifies with DT."""
    if vgg_base_model is None or dt_classifier is None:
        return jsonify({"error": "ML models failed to load on server start. Check console for details."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image file from the request
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Resize image to VGG16 input requirements
        img = img.resize(TARGET_SIZE)
        
        # Convert image to numpy array and prepare for model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) # Add batch dimension
        x = preprocess_input(x) # VGG16-specific preprocessing

        # 1. Feature Extraction (VGG16)
        features = vgg_base_model.predict(x, verbose=0) # Shape: (1, 512)

        # 2. Classification (Decision Tree)
        # We use .predict_proba() to get confidence scores for all classes
        probabilities = dt_classifier.predict_proba(features)
        
        # Get the predicted class index (highest probability)
        predicted_index = np.argmax(probabilities[0])
        
        # Get the predicted label and its confidence
        predicted_shape = CLASS_LABELS[predicted_index]
        confidence = probabilities[0][predicted_index]

        # Return the results as JSON to the React frontend
        return jsonify({
            "predicted_shape": predicted_shape,
            "confidence": f"{confidence:.3f}"
        })

    except Exception as e:
        print(f"Prediction failed due to error: {e}")
        return jsonify({"error": f"Internal prediction error: {e}"}), 500

# --- 4. Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server...")
    # This server must be kept running in its own terminal window
    # The port 5000 is used by the React frontend to communicate
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
