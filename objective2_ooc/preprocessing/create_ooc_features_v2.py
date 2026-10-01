import json
import pickle
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = BASE_DIR / "data" / "merged_balanced"

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "embeddings"
    / "clip_image_embeddings"
)

TEXT_DIR = (
    BASE_DIR
    / "data"
    / "embeddings"
    / "clip_text_embeddings"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pickle(path):

    print(f"Loading: {path.name}")

    with open(path, "rb") as f:
        return pickle.load(f)


def cosine_similarity(image, text):

    image_norm = np.linalg.norm(image)
    text_norm = np.linalg.norm(text)

    return np.dot(image, text) / (
        image_norm * text_norm
    )


def process_split(split):

    print(f"\nProcessing {split}...")

    annotation_path = (
        ANNOTATION_DIR / f"{split}.json"
    )

    image_path = (
        IMAGE_DIR
        / f"clip_image_embeddings_{split}.pkl"
    )

    text_path = (
        TEXT_DIR
        / f"clip_text_embeddings_{split}.pkl"
    )

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    image_embeddings = load_pickle(image_path)
    text_embeddings = load_pickle(text_path)

    features = []
    labels = []

    missing = 0

    for annotation in annotations:

        text_id = annotation["id"]
        image_id = annotation["image_id"]

        if (
            text_id not in text_embeddings
            or image_id not in image_embeddings
        ):
            missing += 1
            continue

        image = np.asarray(
            image_embeddings[image_id],
            dtype=np.float32
        )

        text = np.asarray(
            text_embeddings[text_id],
            dtype=np.float32
        )

        

        absolute_difference = np.abs(
            image - text
        )

        product = image * text

        similarity = cosine_similarity(
            image,
            text
        )

        combined = np.concatenate([
            image,
            text,
            absolute_difference,
            product,
            np.array([similarity], dtype=np.float32)
        ])

        features.append(combined)

        labels.append(
            int(annotation["falsified"])
        )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    print(f"Annotations: {len(annotations)}")
    print(f"Missing embeddings: {missing}")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    print(f"Class 0: {np.sum(y == 0)}")
    print(f"Class 1: {np.sum(y == 1)}")

    np.save(
        OUTPUT_DIR / f"X_{split}.npy",
        X
    )

    np.save(
        OUTPUT_DIR / f"y_{split}.npy",
        y
    )

    print(f"Saved X_{split}.npy")
    print(f"Saved y_{split}.npy")


if __name__ == "__main__":

    for split in ["train", "val", "test"]:

        process_split(split)

    print("\nOOC feature extraction completed successfully.")