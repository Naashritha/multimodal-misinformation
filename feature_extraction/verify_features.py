import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("feature_extraction/features")


SPLITS = {
    "train": {
        "csv": "train_final.csv",
        "text": "train_text_embeddings.npy",
        "image": "train_image_embeddings.npy"
    },
    "val": {
        "csv": "val_final.csv",
        "text": "val_text_embeddings.npy",
        "image": "val_image_embeddings.npy"
    },
    "test": {
        "csv": "test_final.csv",
        "text": "test_text_embeddings.npy",
        "image": "test_image_embeddings.npy"
    }
}


def verify_split(split_name, files):

    print("\n" + "=" * 50)
    print(f"VERIFYING {split_name.upper()}")
    print("=" * 50)

    # Load CSV
    df = pd.read_csv(
        PROCESSED_DIR / files["csv"]
    )

    # Load embeddings
    text_embeddings = np.load(
        FEATURE_DIR / files["text"]
    )

    image_embeddings = np.load(
        FEATURE_DIR / files["image"]
    )

    csv_count = len(df)
    text_count = text_embeddings.shape[0]
    image_count = image_embeddings.shape[0]

    print(f"CSV samples       : {csv_count}")
    print(f"Text embeddings   : {text_count}")
    print(f"Image embeddings  : {image_count}")

    # Check number of samples
    if csv_count != text_count:
        raise ValueError(
            f"Text embedding count mismatch: "
            f"{csv_count} vs {text_count}"
        )

    if csv_count != image_count:
        raise ValueError(
            f"Image embedding count mismatch: "
            f"{csv_count} vs {image_count}"
        )

    # Check dimensions
    print(f"Text feature size  : {text_embeddings.shape[1]}")
    print(f"Image feature size : {image_embeddings.shape[1]}")

    # Check labels
    labels = df["label"].to_numpy()

    if len(labels) != csv_count:
        raise ValueError("Label count mismatch")

    # Check for NaN / infinite values
    if not np.isfinite(text_embeddings).all():
        raise ValueError(
            "Text embeddings contain NaN or infinite values"
        )

    if not np.isfinite(image_embeddings).all():
        raise ValueError(
            "Image embeddings contain NaN or infinite values"
        )

    # Check normalization
    text_norms = np.linalg.norm(
        text_embeddings,
        axis=1
    )

    image_norms = np.linalg.norm(
        image_embeddings,
        axis=1
    )

    print(
        f"Text norm range   : "
        f"{text_norms.min():.4f} - {text_norms.max():.4f}"
    )

    print(
        f"Image norm range  : "
        f"{image_norms.min():.4f} - {image_norms.max():.4f}"
    )

    print(
        f"Class 0           : "
        f"{np.sum(labels == 0)}"
    )

    print(
        f"Class 1           : "
        f"{np.sum(labels == 1)}"
    )

    print("\n✓ CSV and embeddings are aligned")
    print("✓ No NaN/infinite values")
    print("✓ Labels match sample count")


def main():

    for split_name, files in SPLITS.items():
        verify_split(
            split_name,
            files
        )

    print("\n" + "=" * 50)
    print("ALL FEATURE CHECKS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()