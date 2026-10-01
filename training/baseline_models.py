import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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
# Model
# --------------------------------------------------

def create_model():

    return MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42
    )


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

def evaluate(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
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

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# --------------------------------------------------
# Train one experiment
# --------------------------------------------------

def train_experiment(
    name,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 50)
    print(f"Training: {name}")
    print("=" * 50)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = create_model()

    model.fit(
        X_train_scaled,
        y_train
    )

    metrics = evaluate(
        model,
        X_test_scaled,
        y_test,
        name
    )

    return metrics


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading features...")

    (
        train_text,
        train_image,
        train_similarity,
        y_train
    ) = load_split("train")

    (
        val_text,
        val_image,
        val_similarity,
        y_val
    ) = load_split("val")

    (
        test_text,
        test_image,
        test_similarity,
        y_test
    ) = load_split("test")

    results = []

    # --------------------------------------------------
    # 1. Text-only
    # --------------------------------------------------

    results.append(
        train_experiment(
            "Text-only",
            train_text,
            test_text,
            y_train,
            y_test
        )
    )

    # --------------------------------------------------
    # 2. Image-only
    # --------------------------------------------------

    results.append(
        train_experiment(
            "Image-only",
            train_image,
            test_image,
            y_train,
            y_test
        )
    )

    # --------------------------------------------------
    # 3. Text + Image
    # --------------------------------------------------

    train_text_image = np.concatenate(
        [
            train_text,
            train_image
        ],
        axis=1
    )

    test_text_image = np.concatenate(
        [
            test_text,
            test_image
        ],
        axis=1
    )

    results.append(
        train_experiment(
            "Text + Image",
            train_text_image,
            test_text_image,
            y_train,
            y_test
        )
    )

    # --------------------------------------------------
    # 4. Text + Image + Consistency
    # --------------------------------------------------

    train_full = np.concatenate(
        [
            train_text,
            train_image,
            train_similarity.reshape(-1, 1)
        ],
        axis=1
    )

    test_full = np.concatenate(
        [
            test_text,
            test_image,
            test_similarity.reshape(-1, 1)
        ],
        axis=1
    )

    results.append(
        train_experiment(
            "Text + Image + Consistency",
            train_full,
            test_full,
            y_train,
            y_test
        )
    )

    # --------------------------------------------------
    # Save comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = MODEL_DIR / "baseline_comparison.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL BASELINE COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()