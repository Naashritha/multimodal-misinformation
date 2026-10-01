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
# CLIP Model
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
# Extract features
# --------------------------------------------------

def extract_features(csv_name, output_name):

    print("\n" + "=" * 50)
    print(f"Processing {csv_name}")
    print("=" * 50)

    df = pd.read_csv(PROCESSED_DIR / csv_name)

    image_paths = df["image_path"].tolist()

    embeddings = []

    batch_size = 8

    for i in tqdm(
        range(0, len(image_paths), batch_size),
        desc="Extracting image features"
    ):

        batch_paths = image_paths[i:i + batch_size]

        images = []

        for image_path in batch_paths:

            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)

            except Exception as e:
                print(
                    f"\nWarning: Could not open image: {image_path}"
                )

        # Skip batch if no images loaded
        if len(images) == 0:
            continue

        # CLIP preprocessing
        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Extract image features
        with torch.no_grad():

            output = model.get_image_features(
                **inputs
            )

        # Handle Transformers output
        if isinstance(output, torch.Tensor):

            image_features = output

        elif hasattr(output, "pooler_output"):

            image_features = output.pooler_output

        elif hasattr(output, "last_hidden_state"):

            image_features = output.last_hidden_state[:, 0, :]

        else:

            raise TypeError(
                f"Unexpected CLIP output type: {type(output)}"
            )

        # Normalize
        image_features = torch.nn.functional.normalize(
            image_features,
            p=2,
            dim=1
        )

        # Add to list
        embeddings.append(
            image_features.cpu().numpy()
        )

    # Make sure features were actually extracted
    if len(embeddings) == 0:

        raise RuntimeError(
            "No image embeddings were extracted."
        )

    # Combine batches
    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    # Save
    output_path = FEATURE_DIR / output_name

    np.save(
        output_path,
        embeddings
    )

    print("\nSaved:", output_path)
    print("Shape:", embeddings.shape)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    extract_features(
        "train_final.csv",
        "train_image_embeddings.npy"
    )

    extract_features(
        "val_final.csv",
        "val_image_embeddings.npy"
    )

    extract_features(
        "test_final.csv",
        "test_image_embeddings.npy"
    )

    print("\n" + "=" * 50)
    print("IMAGE FEATURE EXTRACTION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()