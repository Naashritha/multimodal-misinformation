import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading test data...")

X_test = np.load("objective2_ooc/data/processed_v2/X_test.npy")
y_test = np.load("objective2_ooc/data/processed_v2/y_test.npy")

print("Test shape:", X_test.shape)

print("\nLoading model...")

model = joblib.load("objective2_ooc/models/ooc_classifier_v2.pkl")
scaler = joblib.load("objective2_ooc/models/ooc_scaler_v2.pkl")

print("Scaling test features...")

X_test_scaled = scaler.transform(X_test)

print("Making predictions...")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n====================================")
print("Objective 2 V2 - Test Evaluation")
print("====================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print("-----------------------------")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print("-----------------------------")
print(classification_report(
    y_test,
    y_pred,
    target_names=["In-Context", "Out-of-Context"]
))