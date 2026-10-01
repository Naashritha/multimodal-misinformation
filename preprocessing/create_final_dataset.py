import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


SPLITS = {
    "train": ("train.csv", "train_final.csv"),
    "val": ("val.csv", "val_final.csv"),
    "test": ("test.csv", "test_final.csv")
}


def create_balanced_dataset(input_file, output_file):

    df = pd.read_csv(PROCESSED_DIR / input_file)

    # Keep only valid images
    df = df[df["valid_image"] == True].copy()

    # Separate classes
    class_0 = df[df["label"] == 0]
    class_1 = df[df["label"] == 1]

    # Balance using the smaller class
    samples_per_class = min(
        len(class_0),
        len(class_1)
    )

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
    final_df.to_csv(
        PROCESSED_DIR / output_file,
        index=False
    )

    print(f"\n{output_file}")
    print("-" * 40)
    print(f"Samples per class : {samples_per_class}")
    print(f"Total samples     : {len(final_df)}")
    print("\nClass distribution:")
    print(final_df["label"].value_counts().sort_index())


def main():

    for split, files in SPLITS.items():

        input_file, output_file = files

        create_balanced_dataset(
            input_file,
            output_file
        )

    print("\n" + "=" * 50)
    print("FINAL BALANCED DATASET CREATED")
    print("=" * 50)


if __name__ == "__main__":
    main()