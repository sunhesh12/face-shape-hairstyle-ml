# src/validation.py
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def validate_model(y_true, y_pred):
    print("\nValidation Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
