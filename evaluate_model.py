import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

TEXT_FEATURES = "feature_extraction/features/test_text_embeddings.npy"
IMAGE_FEATURES = "feature_extraction/features/test_image_embeddings.npy"
CONSISTENCY_FEATURES = "feature_extraction/features/test_clip_similarity.npy"

TEST_CSV = "data/processed/test_final.csv"

MODEL_PATH = "models/multimodal_fusion_model.pkl"
SCALER_PATH = "models/multimodal_scaler.pkl"


# --------------------------------------------------
# Load features
# --------------------------------------------------

print("Loading test features...")

text_features = np.load(TEXT_FEATURES)
image_features = np.load(IMAGE_FEATURES)
consistency_features = np.load(CONSISTENCY_FEATURES)

print("Text features:", text_features.shape)
print("Image features:", image_features.shape)
print("Consistency features:", consistency_features.shape)


# --------------------------------------------------
# Load labels
# --------------------------------------------------

test_df = pd.read_csv(TEST_CSV)

y_true = test_df["label"].astype(int).to_numpy()

print("Labels:", y_true.shape)


# --------------------------------------------------
# Check dimensions
# --------------------------------------------------

assert text_features.shape[0] == len(y_true)
assert image_features.shape[0] == len(y_true)
assert consistency_features.shape[0] == len(y_true)

print("\nAll sample counts match.")


# --------------------------------------------------
# Prepare consistency feature
# --------------------------------------------------

consistency_features = consistency_features.reshape(-1, 1)


# --------------------------------------------------
# Combine features
# --------------------------------------------------

X_test = np.concatenate(
    [
        text_features,
        image_features,
        consistency_features
    ],
    axis=1
)

print("Combined features:", X_test.shape)


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Scale features
# --------------------------------------------------

X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Predict
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MULTIMODAL TEST SET EVALUATION")
print("=" * 60)

print(f"Samples evaluated : {len(y_true)}")
print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Precision         : {precision * 100:.2f}%")
print(f"Recall            : {recall * 100:.2f}%")
print(f"F1 Score          : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Non-Misinformation",
            "Misinformation"
        ],
        digits=4,
        zero_division=0
    )
)