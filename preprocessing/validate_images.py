

import pandas as pd
from pathlib import Path
from PIL import Image

PROCESSED_DIR = Path("data/processed")

SPLITS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}


def validate_split(split_name, csv_name):

    print("\n" + "=" * 50)
    print(f"VALIDATING {split_name.upper()}")
    print("=" * 50)

    csv_path = PROCESSED_DIR / csv_name
    df = pd.read_csv(csv_path)

    valid_images = 0
    invalid_images = 0

    validation_status = []
    validation_error = []

    for i, row in df.iterrows():

        image_path = row["image_path"]

        try:
            if not image_path or not Path(image_path).exists():
                raise FileNotFoundError("Image file not found")

            # Open image
            with Image.open(image_path) as image:

                # Force PIL to actually read the image
                image.load()

                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions")

            validation_status.append(True)
            validation_error.append("")

            valid_images += 1

        except Exception as e:

            validation_status.append(False)
            validation_error.append(str(e))

            invalid_images += 1

    # Add validation information
    df["valid_image"] = validation_status
    df["validation_error"] = validation_error

    # Save updated CSV
    df.to_csv(csv_path, index=False)

    # Only valid images
    valid_df = df[df["valid_image"] == True]

    print(f"Total samples       : {len(df)}")
    print(f"Valid images        : {valid_images}")
    print(f"Invalid/missing     : {invalid_images}")

    print("\nClass distribution among valid images:")

    print(
        valid_df["label"].value_counts().sort_index()
    )


def main():

    for split_name, csv_name in SPLITS.items():
        validate_split(split_name, csv_name)

    print("\n" + "=" * 50)
    print("IMAGE VALIDATION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()