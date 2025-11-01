# src/test.py
import os
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from utils import recommend_hairstyle

# Load models
print("Loading models...")
base_path = os.path.join(os.path.dirname(__file__), "../models")
ft_model = load_model(os.path.join(base_path, "fine_tuned_resnet50.h5"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
pca = joblib.load(os.path.join(base_path, "pca.pkl"))
svm = joblib.load(os.path.join(base_path, "svm_face_shape.pkl"))

# Build feature extractor from ResNet50
feature_extractor = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
feature_extractor.set_weights(ft_model.get_weights()[:len(feature_extractor.get_weights())])

# Predict face shape
def predict_face_shape(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(img_array, verbose=0)
    features = features.reshape(features.shape[0], -1)
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)

    pred = svm.predict(reduced)[0]
    labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    return labels[int(pred)]

# Demo prediction
if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), "../data sheets/zendaya.jpg")
    if os.path.exists(img_path):
        shape = predict_face_shape(img_path)
        styles = recommend_hairstyle(shape)
        print(f"\nPredicted Face Shape: {shape}")
        print("Recommended Hairstyles:")
        for s in styles:
            print("-", s)
    else:
        print(" Please add an image at '../uploads/test_face.jpg' for testing.")
