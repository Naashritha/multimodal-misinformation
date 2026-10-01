import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    return np.sum(a * b, axis=1) / (
        a_norm[:, 0] * b_norm[:, 0]
    )


for split in ["train", "val", "test"]:

    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")

    image_features = X[:, :512]
    text_features = X[:, 512:]

    similarity = cosine_similarity(
        image_features,
        text_features
    )

    print(f"\n===== {split.upper()} =====")

    for label in [0, 1]:

        values = similarity[y == label]

        print(f"\nClass {label}")
        print("Samples :", len(values))
        print("Mean    :", values.mean())
        print("Std     :", values.std())
        print("Min     :", values.min())
        print("Max     :", values.max())

    print("\nOverall")
    print("Mean:", similarity.mean())
    print("Std :", similarity.std())