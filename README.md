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
## 🗂️ Project File Structure

face-shape-hairstyle-ml/
│
├── data sheet/ # Raw and processed datasets
│ ├── hairstyle_prediction_dataset_1000
│ └── synthetic_face_shape_dataset_2000
│
├── notebooks/ # Jupyter notebooks for EDA and model testing
│ ├── DTA_method : FC212001_Heshan Bandaranayake
│ └── fc212008_nathasha/ :J.W.N.F. Silva
│ └── example.ipynb
│
├── models/ # Saved trained models (optional)
│ └── face_shape_model.pkl
│
├── images/ # Sample input/output or dataset visualizations
│ └── test_image.jpg
│
├── app.py # Streamlit web app file
├── train_model.py # Script to train ML models
├── predict.py # Face shape and hairstyle prediction script
├── requirements.txt # Python dependencies
└── README.md # Project description (this file)


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


| Name                                                                                                                        | FC Number           | Role                                      |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------- |
| **H.S.D.B. Bandaranayake**                                                                                                  | FC110573 / FC212001 | 🧠 **Model Lead**                         |
| Leads the development of machine learning models, selects algorithms, trains classifiers, and fine-tunes hyperparameters.   |                     |                                           |
|                                                                                                                             |                     |                                           |
| **N.F.M.A. Fernandopulle**                                                                                                  | FC110611 / FC212038 | 🧹 **Data Handling & Preprocessing Lead** |
| Responsible for dataset collection, cleaning, facial feature extraction, and formatting data for training.                  |                     |                                           |
|                                                                                                                             |                     |                                           |
| **K.A.G. Kawya Thathsarani**                                                                                                | FC110584 / FC212011 | 🎨 **UI & Deployment Lead**               |
| Designs and implements the user interface using Streamlit or Flask. Integrates model predictions into a usable frontend.    |                     |                                           |
|                                                                                                                             |                     |                                           |
| **J.W\.N.F. Silva**                                                                                                         | FC110580 / FC212008 | 📊 **Testing & Evaluation Specialist**    |
| Conducts model testing, evaluates accuracy and performance metrics, and validates output consistency.                       |                     |                                           |
|                                                                                                                             |                     |                                           |
| **W\.K.I. Sathsarani**                                                                                                      | FC110903 / FC212044 | 📚 **Documentation & Research Lead**      |
| Prepares the proposal, README, final documentation, and researches hairstyle-face shape mappings and supporting literature. |                     |                                           |
|                                                                                                                             |                     |                                           |


```
