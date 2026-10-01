import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load datasets
# -----------------------------

print("Loading datasets...")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# -----------------------------
# Standardization
# -----------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# MLP Classifier
# -----------------------------

print("\nTraining OOC classifier...")

model = MLPClassifier(
    hidden_layer_sizes=(512, 128),
    activation="relu",
    solver="adam",
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)


# -----------------------------
# Evaluation function
# -----------------------------

def evaluate(name, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print(f"\n{name} Results")
    print("-----------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return accuracy, precision, recall, f1


# -----------------------------
# Evaluate
# -----------------------------

val_metrics = evaluate(
    "Validation",
    X_val_scaled,
    y_val
)

test_metrics = evaluate(
    "Test",
    X_test_scaled,
    y_test
)


# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    model,
    MODEL_DIR / "ooc_classifier.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "ooc_scaler.pkl"
)


# -----------------------------
# Save metrics
# -----------------------------

with open(MODEL_DIR / "ooc_metrics.csv", "w") as f:

    f.write("split,accuracy,precision,recall,f1\n")

    f.write(
        f"validation,"
        f"{val_metrics[0]},"
        f"{val_metrics[1]},"
        f"{val_metrics[2]},"
        f"{val_metrics[3]}\n"
    )

    f.write(
        f"test,"
        f"{test_metrics[0]},"
        f"{test_metrics[1]},"
        f"{test_metrics[2]},"
        f"{test_metrics[3]}\n"
    )


print("\nModel saved successfully.")
print("Training completed.")