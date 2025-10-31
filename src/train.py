# src/train.py
import os
import joblib
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import ImageFile
from utils import load_data, extract_features

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================
# Step 1 — Load Dataset
# ==============================
print("Loading dataset...")
data_dir = "../data sheets/training_set"
train_gen, val_gen = load_data(data_dir)

# ==============================
# Step 2 — Fine-tune ResNet50
# ==============================
print("Building and fine-tuning ResNet50...")
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(5, activation='softmax')(x)

ft_model = Model(inputs=base_model.input, outputs=predictions)
ft_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

ft_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# Step 3 — Extract Features
# ==============================
print("Extracting features...")
feature_extractor = Model(inputs=ft_model.input, outputs=ft_model.get_layer("conv5_block3_out").output)
X_train, y_train = extract_features(train_gen, feature_extractor)
X_test, y_test = extract_features(val_gen, feature_extractor)

# ==============================
# Step 4 — Normalize + PCA
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=256, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ==============================
# Step 5 — Train + Grid Search SVM
# ==============================
print("Training SVM with GridSearchCV...")
param_grid = {
    'C': [1, 5, 10, 50],
    'gamma': ['scale', 0.001, 0.0005],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

clf = grid.best_estimator_
clf.fit(X_train, y_train)

# ==============================
# Step 6 — Evaluate
# ==============================
y_pred = clf.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# Step 7 — Save Models
# ==============================
os.makedirs("../models", exist_ok=True)
ft_model.save("../models/fine_tuned_resnet50.h5")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(pca, "../models/pca.pkl")
joblib.dump(clf, "../models/svm_face_shape.pkl")

print("Models saved successfully in '../models/' directory.")
