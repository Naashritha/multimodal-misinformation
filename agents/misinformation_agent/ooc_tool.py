import os
import joblib
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "objective2_ooc",
    "models",
    "ooc_classifier_v2.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "objective2_ooc",
    "models",
    "ooc_scaler_v2.pkl"
)


# --------------------------------------------------
# Load OOC classifier and scaler
# --------------------------------------------------

ooc_model = joblib.load(MODEL_PATH)
ooc_scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Load CLIP
# --------------------------------------------------

device = "cpu"

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()


# --------------------------------------------------
# OOC Prediction Tool
# --------------------------------------------------

def predict_ooc(
    text: str,
    image_path: str
) -> dict:
    """
    Predict whether the image is used in-context
    or out-of-context with the given text.
    """

    try:

        # ------------------------------------------
        # Load image
        # ------------------------------------------

        from PIL import Image

        image = Image.open(image_path).convert("RGB")


        # ------------------------------------------
        # CLIP feature extraction
        # ------------------------------------------

        inputs = clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        with torch.no_grad():

            image_output = clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            image_features = clip_model.visual_projection(
                image_output.pooler_output
            )

            text_output = clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            text_features = clip_model.text_projection(
                text_output.pooler_output
            )



        # ------------------------------------------
        # Normalize embeddings
        # ------------------------------------------

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )


        image_embedding = image_features.cpu().numpy()[0]

        text_embedding = text_features.cpu().numpy()[0]


        # ------------------------------------------
        # Construct 2049-D feature vector
        # ------------------------------------------

        difference = np.abs(
            image_embedding - text_embedding
        )

        product = (
            image_embedding * text_embedding
        )

        cosine_similarity = float(
            np.dot(
                image_embedding,
                text_embedding
            )
        )


        features = np.concatenate([
            image_embedding,
            text_embedding,
            difference,
            product,
            [cosine_similarity]
        ])


        # ------------------------------------------
        # Scale features
        # ------------------------------------------

        features = features.reshape(1, -1)

        features_scaled = ooc_scaler.transform(
            features
        )


        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = int(
            ooc_model.predict(features_scaled)[0]
        )

        probabilities = ooc_model.predict_proba(
            features_scaled
        )[0]


        # NewsCLIPpings labels:
        # 0 = In-Context
        # 1 = Out-of-Context

        if prediction == 1:
            label = "OUT-OF-CONTEXT"
        else:
            label = "IN-CONTEXT"


        confidence = float(
            probabilities[prediction]
        )


        return {
            "status": "success",
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "cosine_similarity": round(
                cosine_similarity,
                4
            )
        }


    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }