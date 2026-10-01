import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

for split in ["train", "val", "test"]:
    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")

    print(f"\n===== {split.upper()} =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Class 0:", np.sum(y == 0))
    print("Class 1:", np.sum(y == 1))

    print("Feature min:", X.min())
    print("Feature max:", X.max())
    print("Feature mean:", X.mean())
    print("Feature std:", X.std())

    # Image and text portions
    image_features = X[:, :512]
    text_features = X[:, 512:]

    print("\nImage features:")
    print("  mean:", image_features.mean())
    print("  std :", image_features.std())

    print("Text features:")
    print("  mean:", text_features.mean())
    print("  std :", text_features.std())

    # Check whether samples have meaningful variation
    print("\nFirst 5 labels:", y[:5])
    print("First sample first 10 features:", X[0][:10])