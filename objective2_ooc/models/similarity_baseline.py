import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def calculate_similarity(X):
    image = X[:, :512]
    text = X[:, 512:]

    image_norm = np.linalg.norm(image, axis=1)
    text_norm = np.linalg.norm(text, axis=1)

    similarity = np.sum(image * text, axis=1) / (
        image_norm * text_norm
    )

    return similarity


X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


train_similarity = calculate_similarity(X_train)
val_similarity = calculate_similarity(X_val)
test_similarity = calculate_similarity(X_test)


# Find threshold using training data.
# Lower similarity indicates a potentially falsified/OOC pair.
thresholds = np.linspace(
    train_similarity.min(),
    train_similarity.max(),
    500
)

best_threshold = None
best_accuracy = -1

for threshold in thresholds:

    predictions = (
        train_similarity < threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_train,
        predictions
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold


def evaluate(name, similarity, y):

    predictions = (
        similarity < best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions
    )

    recall = recall_score(
        y,
        predictions
    )

    f1 = f1_score(
        y,
        predictions
    )

    print(f"\n{name} Results")
    print("-----------------------------")
    print(f"Threshold : {best_threshold:.6f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


print("Best training threshold:", best_threshold)

evaluate(
    "Validation",
    val_similarity,
    y_val
)

evaluate(
    "Test",
    test_similarity,
    y_test
)   