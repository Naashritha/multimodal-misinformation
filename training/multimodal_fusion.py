import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("feature_extraction/features")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_split(split):

    df = pd.read_csv(
        PROCESSED_DIR / f"{split}_final.csv"
    )

    text = np.load(
        FEATURE_DIR / f"{split}_text_embeddings.npy"
    )

    image = np.load(
        FEATURE_DIR / f"{split}_image_embeddings.npy"
    )

    similarity = np.load(
        FEATURE_DIR / f"{split}_clip_similarity.npy"
    )

    labels = df["label"].to_numpy()

    return text, image, similarity, labels


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def evaluate_model(model, X, y, name):

    predictions = model.predict(X)

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

    cm = confusion_matrix(
        y,
        predictions
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Class 0",
                "Class 1"
            ]
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# --------------------------------------------------
# Train MLP
# --------------------------------------------------

def create_model():

    return MLPClassifier(
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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading features...")

    train_text, train_image, train_similarity, y_train = \
        load_split("train")

    val_text, val_image, val_similarity, y_val = \
        load_split("val")

    test_text, test_image, test_similarity, y_test = \
        load_split("test")

    print("\nFeature shapes:")

    print("Train text      :", train_text.shape)
    print("Train image     :", train_image.shape)
    print("Train similarity:", train_similarity.shape)

    # --------------------------------------------------
    # Multimodal fusion
    # --------------------------------------------------

    X_train = np.concatenate(
        [
            train_text,
            train_image,
            train_similarity.reshape(-1, 1)
        ],
        axis=1
    )

    X_val = np.concatenate(
        [
            val_text,
            val_image,
            val_similarity.reshape(-1, 1)
        ],
        axis=1
    )

    X_test = np.concatenate(
        [
            test_text,
            test_image,
            test_similarity.reshape(-1, 1)
        ],
        axis=1
    )

    print("\nFusion feature shapes:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nTraining multimodal classifier...")

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_metrics = evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION RESULTS"
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST RESULTS"
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    joblib.dump(
        model,
        MODEL_DIR / "multimodal_fusion_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "multimodal_scaler.pkl"
    )

    print("\nModels saved:")

    print(
        "models/multimodal_fusion_model.pkl"
    )

    print(
        "models/multimodal_scaler.pkl"
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "split": "validation",
                **val_metrics
            },
            {
                "split": "test",
                **test_metrics
            }
        ]
    )

    metrics.to_csv(
        MODEL_DIR / "multimodal_metrics.csv",
        index=False
    )

    print(
        "\nSaved: models/multimodal_metrics.csv"
    )

    print("\n" + "=" * 50)
    print("MULTIMODAL TRAINING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()