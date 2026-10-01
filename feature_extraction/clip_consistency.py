import pandas as pd
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("feature_extraction/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CLIP
# --------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP processor...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)

device = torch.device("cpu")
model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False


# --------------------------------------------------
# Calculate text-image consistency
# --------------------------------------------------

def calculate_consistency(csv_name, output_name):

    print("\n" + "=" * 50)
    print(f"Processing {csv_name}")
    print("=" * 50)

    df = pd.read_csv(
        PROCESSED_DIR / csv_name
    )

    texts = df["text"].fillna("").astype(str).tolist()
    image_paths = df["image_path"].tolist()

    similarity_scores = []

    batch_size = 8

    for i in tqdm(
        range(0, len(df), batch_size),
        desc="Calculating CLIP consistency"
    ):

        batch_texts = texts[i:i + batch_size]
        batch_paths = image_paths[i:i + batch_size]

        valid_texts = []
        valid_images = []

        for text, image_path in zip(
            batch_texts,
            batch_paths
        ):

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                valid_texts.append(text)
                valid_images.append(image)

            except Exception:

                # This should normally not happen
                # because we already validated images.
                continue

        if len(valid_images) == 0:
            continue

        # Process text and image together
        inputs = processor(
            text=valid_texts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            # CLIP text embeddings
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            # CLIP image embeddings
            image_features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

        # Handle Transformers output
        if not isinstance(text_features, torch.Tensor):

            if hasattr(
                text_features,
                "pooler_output"
            ):
                text_features = (
                    text_features.pooler_output
                )

            elif hasattr(
                text_features,
                "last_hidden_state"
            ):
                text_features = (
                    text_features.last_hidden_state[:, 0, :]
                )

        if not isinstance(image_features, torch.Tensor):

            if hasattr(
                image_features,
                "pooler_output"
            ):
                image_features = (
                    image_features.pooler_output
                )

            elif hasattr(
                image_features,
                "last_hidden_state"
            ):
                image_features = (
                    image_features.last_hidden_state[:, 0, :]
                )

        # Normalize
        text_features = torch.nn.functional.normalize(
            text_features,
            p=2,
            dim=1
        )

        image_features = torch.nn.functional.normalize(
            image_features,
            p=2,
            dim=1
        )

        # Cosine similarity
        similarity = torch.sum(
            text_features * image_features,
            dim=1
        )

        similarity_scores.extend(
            similarity.cpu().numpy().tolist()
        )

    similarity_scores = np.array(
        similarity_scores,
        dtype=np.float32
    )

    # Safety check
    if len(similarity_scores) != len(df):

        raise RuntimeError(
            f"Expected {len(df)} similarity scores "
            f"but got {len(similarity_scores)}"
        )

    output_path = FEATURE_DIR / output_name

    np.save(
        output_path,
        similarity_scores
    )

    print("\nSaved:", output_path)

    print(
        "Shape:",
        similarity_scores.shape
    )

    print(
        f"Min similarity: "
        f"{similarity_scores.min():.4f}"
    )

    print(
        f"Max similarity: "
        f"{similarity_scores.max():.4f}"
    )

    print(
        f"Mean similarity: "
        f"{similarity_scores.mean():.4f}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    calculate_consistency(
        "train_final.csv",
        "train_clip_similarity.npy"
    )

    calculate_consistency(
        "val_final.csv",
        "val_clip_similarity.npy"
    )

    calculate_consistency(
        "test_final.csv",
        "test_clip_similarity.npy"
    )

    print("\n" + "=" * 50)
    print("CLIP CONSISTENCY EXTRACTION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()