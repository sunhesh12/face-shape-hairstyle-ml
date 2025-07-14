import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import joblib
import pandas as pd

# Load DTA model and scaler
model = joblib.load(r"D:\education\japura\3rd year\1st sem\Machine Learning\project\HairstylePredicAccordingToFaceshape\notebooks\DTA_method\DTA_Model.pkl")
scaler = joblib.load(r"D:\education\japura\3rd year\1st sem\Machine Learning\project\HairstylePredicAccordingToFaceshape\notebooks\DTA_method\Scaler.pkl")

# Load dataset and map hairstyles
df = pd.read_csv("synthetic_face_shape_dataset_5000.csv")
df.columns = df.columns.str.strip()
face_to_hairstyle = df.drop_duplicates(subset=['face shape'])[['face shape', 'best hairstyle']].set_index('face shape')['best hairstyle'].to_dict()

# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh

def extract_face_features(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Key points
            left_forehead = 10
            right_forehead = 338
            left_cheek = 234
            right_cheek = 454
            chin = 152
            nose_tip = 1

            forehead_width = np.linalg.norm(np.array(points[left_forehead]) - np.array(points[right_forehead]))
            face_width = np.linalg.norm(np.array(points[left_cheek]) - np.array(points[right_cheek]))
            chin_length = np.linalg.norm(np.array(points[nose_tip]) - np.array(points[chin]))
            face_height = np.linalg.norm(np.array(points[10]) - np.array(points[152]))  # top to chin

            return [forehead_width, face_width, face_height, chin_length], results.multi_face_landmarks[0]
    return None, None

# Streamlit app UI
st.markdown("<h2 style='color:#4CAF50;'>Smart Hairstyle Recommender (DTA Model)</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a front-facing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    features, landmarks = extract_face_features(img_cv)
    if features:
        # Scale and predict
        input_df = pd.DataFrame([features], columns=['forehead width (cm)', 'jaw width (cm)', 'face length (cm)', 'face width (cm)'])
        scaled_features = scaler.transform(input_df)
        face_shape = model.predict(scaled_features)[0]

        # Hairstyle recommendation
        hairstyle = face_to_hairstyle.get(face_shape, "No recommendation")

        # Show results
        st.markdown(f"<h4 style='color:#2196F3;'>Predicted Face Shape: <b>{face_shape}</b></h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#9C27B0;'>Recommended Hairstyle: <b>{hairstyle}</b></h4>", unsafe_allow_html=True)

    else:
        st.error("Face not detected. Please upload a clear front-facing image.")
