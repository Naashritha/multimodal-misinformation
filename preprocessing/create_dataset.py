import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data/raw/fakeddit")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_split(file_name, output_name, samples_per_class):
    print(f"\nProcessing {file_name}...")

    df = pd.read_csv(DATA_DIR / file_name, sep="\t")

    # Keep only samples having text and image
    df = df[
        df["clean_title"].notna()
        & df["image_url"].notna()
        & (df["hasImage"] == True)
    ]

    # Select required columns
    df = df[
        ["id", "clean_title", "image_url", "2_way_label"]
    ].copy()

    # Rename columns
    df.rename(
        columns={
            "clean_title": "text",
            "2_way_label": "label"
        },
        inplace=True
    )

    # Remove duplicate IDs
    df.drop_duplicates(subset=["id"], inplace=True)

    # Make sure both classes are available
    class_0 = df[df["label"] == 0]
    class_1 = df[df["label"] == 1]

    if len(class_0) < samples_per_class:
        raise ValueError(
            f"Not enough class 0 samples: {len(class_0)}"
        )

    if len(class_1) < samples_per_class:
        raise ValueError(
            f"Not enough class 1 samples: {len(class_1)}"
        )

    # Balanced sampling
    class_0 = class_0.sample(
        n=samples_per_class,
        random_state=42
    )

    class_1 = class_1.sample(
        n=samples_per_class,
        random_state=42
    )

    # Combine
    final_df = pd.concat(
        [class_0, class_1],
        ignore_index=True
    )

    # Shuffle
    final_df = final_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save
    output_path = OUTPUT_DIR / output_name
    final_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Samples: {len(final_df)}")
    print("Labels:")
    print(final_df["label"].value_counts())


# 16,000 training samples
prepare_split(
    "all_train.tsv",
    "train.csv",
    8000
)

# 2,000 validation samples
prepare_split(
    "all_validate.tsv",
    "val.csv",
    1000
)

# 2,000 test samples
prepare_split(
    "all_test_public.tsv",
    "test.csv",
    1000
)

print("\n==============================")
print("20K DATASET CREATED")
print("==============================")
print("Train      : 16,000")
print("Validation : 2,000")
print("Test       : 2,000")
print("Total      : 20,000")