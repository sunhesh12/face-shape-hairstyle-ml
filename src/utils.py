# src/utils.py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# ==============================
# Data Loading Utility
# ==============================
def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen


# ==============================
# Feature Extraction Utility
# ==============================
def extract_features(generator, model, batch_size=32):
    features, labels = [], []
    for i in range(len(generator)):
        try:
            x_batch, y_batch = generator[i]
            feat_batch = model.predict(x_batch, verbose=0)
            feat_batch = feat_batch.reshape(feat_batch.shape[0], -1)
            features.append(feat_batch)
            labels.append(y_batch)
            if (i + 1) * batch_size >= generator.n:
                break
        except Exception as e:
            print(f"Skipping batch {i} due to error: {e}")
            continue

    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y


# ==============================
# Hairstyle Recommendation
# ==============================
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
