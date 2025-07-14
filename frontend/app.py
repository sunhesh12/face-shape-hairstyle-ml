import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import joblib
import pandas as pd
from collections import defaultdict

# Load dataset for rule-based system
df = pd.read_csv("D:/5th Sem/Machine Learning/ML Project/HairstylePredicAccordingToFaceshape/data sheets/synthetic_face_shape_dataset_5000.csv")
df.columns = df.columns.str.strip() 


hairstyle_dict = {}
for face_shape, group in df.groupby("face shape"):
    top_hairstyles = group["best hairstyle"].value_counts().nlargest(3).index.tolist()
    hairstyle_dict[face_shape.lower()] = top_hairstyles


#Load the trained face shape classifier
model = joblib.load('model') #Add the selected model path here

#MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)



def extract_face_features(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

         # Convert normalized coordinates to pixel coordinates
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Define key landmark indices
        left_forehead = 10        # Approx. top center of forehead
        right_forehead = 338      # Approx. right side
        left_cheek = 234          # Left cheek
        right_cheek = 454         # Right cheek
        jaw_left = 234            # Left jaw
        jaw_right = 454           # Right jaw
        chin = 152                # Chin bottom
        nose_tip = 1              # Nose tip or top of the nose bridge

        # Measure distances
        forehead_width = np.linalg.norm(np.array(points[left_forehead]) - np.array(points[right_forehead]))
        face_width = np.linalg.norm(np.array(points[left_cheek]) - np.array(points[right_cheek]))
        jaw_width = np.linalg.norm(np.array(points[jaw_left]) - np.array(points[jaw_right]))
        chin_length = np.linalg.norm(np.array(points[nose_tip]) - np.array(points[chin]))

        
         # Draw landmarks for debug (optional in Streamlit)
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
        
        # Convert BGR to RGB for displaying in Streamlit
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Facial Landmarks", width=300)
       
        # Normalize features by face width to keep it resolution-independent
        if face_width != 0:
            features = [
                forehead_width / face_width,
                jaw_width / face_width,
                chin_length / face_width,
                face_width
            ]
            return features
        else:
            return None
        
        
    else:
        return None
    
#Streamlit UI
st.markdown("<h2 style='color:#4CAF50;'>Smart Hairstyle Recommender </h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an front-facing image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image', width=300)

    #Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    features = extract_face_features(img_cv)
    if features:
        face_shape = model.predict([features])[0]
        hairstyles = hairstyle_dict.get(face_shape.lower(), ["No recommendation available"])
        st.markdown(f"<h4 style='color:#2196F3;'> Predicted Face Shape: <b>{face_shape.capitalize()}</b></h4>", unsafe_allow_html=True)

        st.markdown("<h4 style='color:#9C27B0;'> Top  Recommended Hairstyles:</h4>", unsafe_allow_html=True)
        for style in hairstyles:
         st.markdown(f"<li style='color:white;'>{style}</li>",     unsafe_allow_html=True)
        


       
    else:
        st.error("No face detected in the image. Please upload a clear image with a visible front-facing image face.")