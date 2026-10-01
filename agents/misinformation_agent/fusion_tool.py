import os
import joblib
import numpy as np

from google.adk.tools import ToolContext


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "multimodal_fusion_model.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "multimodal_scaler.pkl"
)


fusion_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_misinformation(
    tool_context: ToolContext
) -> dict:
    """
    Predict misinformation using the text embedding,
    image embedding, and consistency score stored
    in ADK session state.
    """

    # Get stored embeddings
    text_embedding = tool_context.state.get("text_embedding")
    image_embedding = tool_context.state.get("image_embedding")
    consistency_score = tool_context.state.get("consistency_score")

    # Check that all required features are available
    if text_embedding is None:
        return {
            "status": "error",
            "message": "Text embedding is not available."
        }

    if image_embedding is None:
        return {
            "status": "error",
            "message": "Image embedding is not available."
        }

    if consistency_score is None:
        return {
            "status": "error",
            "message": "Consistency score is not available."
        }

    # Convert to numpy arrays
    text_embedding = np.array(
        text_embedding,
        dtype=np.float32
    )

    image_embedding = np.array(
        image_embedding,
        dtype=np.float32
    )

    consistency = np.array(
        [consistency_score],
        dtype=np.float32
    )

    # Expected dimensions:
    # Text = 384
    # Image = 512
    # Consistency = 1
    # Total = 897

    combined_features = np.concatenate([
        text_embedding,
        image_embedding,
        consistency
    ])

    combined_features = combined_features.reshape(1, -1)

    # Safety check
    if combined_features.shape[1] != 897:
        return {
            "status": "error",
            "message": (
                f"Expected 897 features, "
                f"but received {combined_features.shape[1]}."
            )
        }

    # Scale features
    scaled_features = scaler.transform(
        combined_features
    )

    # Prediction
    prediction = fusion_model.predict(
        scaled_features
    )[0]

    probability = fusion_model.predict_proba(
        scaled_features
    )[0]

    return {
        "status": "success",
        "prediction": int(prediction),
        "label": (
            "MISINFORMATION"
            if prediction == 1
            else "NON-MISINFORMATION"
        ),
        "confidence": float(max(probability)),
        "consistency_score": float(consistency_score),
        "feature_dimension": int(combined_features.shape[1])
    }