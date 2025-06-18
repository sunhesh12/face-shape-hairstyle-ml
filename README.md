# 👤💇 Face Shape Detection and Hairstyle Recommendation System

This project uses traditional Machine Learning (not deep learning) to detect the **face shape** of a person from an image and recommends suitable **hairstyles** based on that shape.

---

## 🚀 Features

- 📸 Accepts a front-facing image of a person.
- 🔍 Extracts facial landmarks using MediaPipe.
- 📐 Computes geometric face features (width, height, jaw angle, etc.).
- 🧠 Trains multiple ML models to classify face shape.
- 💇 Recommends top hairstyles for each predicted shape.
- 🌐 Optional Streamlit interface for live prediction and UI.

---

## 🧑‍💻 Technologies Used

- Python
- Scikit-learn (ML)
- Pandas, NumPy (data handling)
- MediaPipe / Dlib (landmark detection)
- Matplotlib / Seaborn (visualization)
- Streamlit (optional UI)

---

## 📁 Dataset

### 1. Face Shape Dataset
- Format: CSV + images
- Features: facial dimensions (jaw width, cheekbone width, eye distance, etc.)
- Labels: `face_shape` (e.g., "oval", "round", "square", etc.)

### 2. Hairstyle Mapping
- A dictionary or CSV file mapping each face shape to a list of ideal hairstyles.

> You can use:
> - Roboflow face shape datasets  
> - Mediapipe for landmark extraction  
> - K-Hairstyle dataset for hairstyle references

---

## 🛠️ Installation

```bash
git clone https://github.com/sunhesh12/face-shape-hairstyle-ml.git
cd face-shape-hairstyle-ml
pip install -r requirements.txt
````

---

## 🧪 Training the Model

```python
# Load dataset
python train_model.py
```

This script:

* Loads your CSV data
* Extracts features
* Trains models: Random Forest, SVM, k-NN, etc.
* Evaluates performance using accuracy, precision, and confusion matrix

---

## 🧠 Predict Face Shape + Recommend Hairstyles

```python
from predict import predict_face_shape_and_hairstyles

predict_face_shape_and_hairstyles("test_images/user1.jpg")
```

---

## 🌐 Streamlit Web App (Optional)

```bash
streamlit run app.py
```

Features:

* Upload image
* View predicted face shape
* View recommended hairstyles

---

## 🗓️ Project Timeline

| Week | Task                                   |
| ---- | -------------------------------------- |
| 1    | Data collection and feature extraction |
| 2    | Model training (5 algorithms)          |
| 3    | Evaluation and tuning                  |
| 4    | UI deployment and documentation        |

---

## 📊 Evaluation Metrics

* Accuracy
* Precision / Recall
* Confusion Matrix
* Cross-validation scores

---

## ⚠️ Challenges

* Facial landmark noise
* Lack of diverse datasets
* Class imbalance between face shapes
* Hair occlusion / blur in input images

---

## 📚 References

* [Mediapipe by Google](https://google.github.io/mediapipe/)
* [Roboflow Datasets](https://roboflow.com/)
* [K-Hairstyle Dataset](https://github.com/khairstyle-dataset)
* [Scikit-learn Documentation](https://scikit-learn.org/)

---

## 👨‍🎓 Team

* FC110573 - Heshan Bandaranayake (Model & UI)
* FCXXXXX - \[Other Member Name] (Data, Feature Engineering)

```

---

Would you like me to generate `train_model.py`, `predict.py`, or `app.py` files for this project?
```
