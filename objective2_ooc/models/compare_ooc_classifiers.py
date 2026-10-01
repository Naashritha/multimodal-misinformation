import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Loading datasets...")

X_train = np.load("objective2_ooc/data/processed_v2/X_train.npy")
y_train = np.load("objective2_ooc/data/processed_v2/y_train.npy")

X_val = np.load("objective2_ooc/data/processed_v2/X_val.npy")
y_val = np.load("objective2_ooc/data/processed_v2/y_val.npy")

X_test = np.load("objective2_ooc/data/processed_v2/X_test.npy")
y_test = np.load("objective2_ooc/data/processed_v2/y_test.npy")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\nEvaluating...")

y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)


def evaluate(y_true, y_pred, name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name}")
    print("-----------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return accuracy, precision, recall, f1


val_metrics = evaluate(
    y_val,
    y_val_pred,
    "Validation Results"
)

test_metrics = evaluate(
    y_test,
    y_test_pred,
    "Test Results"
)

print("\n====================================")
print("Classifier Comparison")
print("====================================")

print("\nMLP V2:")
print("Accuracy : 70.29%")
print("F1 Score : 68.03%")

print("\nLogistic Regression:")
print(f"Accuracy : {test_metrics[0] * 100:.2f}%")
print(f"F1 Score : {test_metrics[3] * 100:.2f}%")

if test_metrics[0] > 0.7029:
    print("\nLogistic Regression performed better than MLP V2.")
else:
    print("\nMLP V2 performed better than Logistic Regression.")

print("\nSaving Logistic Regression model...")

joblib.dump(
    model,
    "objective2_ooc/models/ooc_classifier_lr.pkl"
)

joblib.dump(
    scaler,
    "objective2_ooc/models/ooc_scaler_lr.pkl"
)

print("Model saved successfully.")