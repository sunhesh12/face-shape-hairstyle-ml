import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import joblib
import pandas as pd

# ----------------------------------------------
# 1. Load hairstyle recommendation dataset
# ----------------------------------------------
df = pd.read_csv("synthetic_face_shape_dataset_5000.csv")
df.columns = df.columns.str.strip()

hairstyle_dict = {}
for face_shape, group in df.groupby("face shape"):
    top_hairstyles = group["best hairstyle"].value_counts().nlargest(3).index.tolist()
    hairstyle_dict[face_shape.lower()] = top_hairstyles

# ----------------------------------------------
# 2. Load the trained SVM face shape model
# ----------------------------------------------
model = joblib.load(
    r'D:\education\japura\3rd year\1st sem\Machine Learning\project\HairstylePredicAccordingToFaceshape\notebooks\SVM_Classification\SVM_Model.pkl'
)

# ----------------------------------------------
# 3. Facial feature extraction using MediaPipe
# ----------------------------------------------
# def extract_face_features(image):
#     mp_face_mesh = mp.solutions.face_mesh
#     with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark
#             h, w, _ = image.shape
#             points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

#             # Key landmarks
#             left_forehead = 10
#             right_forehead = 338
#             left_cheek = 234
#             right_cheek = 454
#             jaw_left = 234
#             jaw_right = 454
#             chin = 152
#             nose_tip = 1

#             forehead_width = np.linalg.norm(np.array(points[left_forehead]) - np.array(points[right_forehead]))
#             face_width = np.linalg.norm(np.array(points[left_cheek]) - np.array(points[right_cheek]))
#             jaw_width = np.linalg.norm(np.array(points[jaw_left]) - np.array(points[jaw_right]))
#             chin_length = np.linalg.norm(np.array(points[nose_tip]) - np.array(points[chin]))

#             # Normalize features by face width
#             if face_width != 0:
#                 features = [
#                     forehead_width / face_width,
#                     jaw_width / face_width,
#                     chin_length / face_width,
#                     face_width
#                 ]
#                 return features, results.multi_face_landmarks[0]
#     return None, None

def extract_face_features(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Landmark indices
            left_forehead = 10
            right_forehead = 338
            left_cheek = 234
            right_cheek = 454
            jaw_left = 234
            jaw_right = 454
            chin = 152
            nose_tip = 1

            forehead_width = np.linalg.norm(np.array(points[left_forehead]) - np.array(points[right_forehead]))
            face_width = np.linalg.norm(np.array(points[left_cheek]) - np.array(points[right_cheek]))
            jaw_width = np.linalg.norm(np.array(points[jaw_left]) - np.array(points[jaw_right]))
            chin_length = np.linalg.norm(np.array(points[nose_tip]) - np.array(points[chin]))

            if face_width != 0:
                features = [
                    forehead_width / face_width,
                    jaw_width / face_width,
                    chin_length / face_width,
                    face_width
                ]
                return features, results.multi_face_landmarks[0]
    return None, None


# ----------------------------------------------
# 4. Streamlit Web UI
# ----------------------------------------------
st.set_page_config(page_title="Smart Hairstyle Recommender")
st.markdown("<h2 style='color:#4CAF50;'>💇‍♀️ Smart Hairstyle Recommender</h2>", unsafe_allow_html=True)
st.info("Upload a clear, front-facing image to get personalized hairstyle suggestions.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Convert PIL to OpenCV image
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Feature extraction
    features, landmarks = extract_face_features(img_cv)

    if features:
        # Predict face shape
        face_shape = model.predict([features])[0]
        hairstyles = hairstyle_dict.get(face_shape.lower(), ["No recommendation available"])

        st.markdown(f"<h4 style='color:#2196F3;'>Predicted Face Shape: <b>{face_shape.capitalize()}</b></h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#9C27B0;'>Top Recommended Hairstyles:</h4>", unsafe_allow_html=True)
        for style in hairstyles:
            st.markdown(f"<li style='color:white;'>{style}</li>", unsafe_allow_html=True)

        # Optional: Draw facial landmarks
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = img_cv.copy()
        mp_drawing.draw_landmarks(annotated_image, landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="Facial Landmarks", width=300)

    else:
        st.error("😕 No face detected. Please upload a clear, front-facing image.")
