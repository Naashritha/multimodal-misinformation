import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# Paths
PROCESSED_DIR = Path("data/processed")
IMAGE_DIR = Path("data/images")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}


def download_image(row, output_dir):
    image_id = str(row["id"])
    image_url = row["image_url"]

    output_path = output_dir / f"{image_id}.jpg"

    # Reuse existing image
    if output_path.exists():
        return image_id, str(output_path), "already_exists", ""

    try:
        response = requests.get(
            image_url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        )

        # Convert to RGB and save as JPEG
        image = image.convert("RGB")
        image.save(
            output_path,
            format="JPEG",
            quality=90
        )

        return image_id, str(output_path), "success", ""

    except Exception as e:
        return image_id, "", "failed", str(e)


def process_split(split_name, csv_name):

    print("\n" + "=" * 50)
    print(f"Processing {split_name.upper()}")
    print("=" * 50)

    csv_path = PROCESSED_DIR / csv_name
    image_dir = IMAGE_DIR / split_name

    image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Add columns if they don't exist
    if "image_path" not in df.columns:
        df["image_path"] = ""

    if "status" not in df.columns:
        df["status"] = ""

    if "error" not in df.columns:
        df["error"] = ""

    rows = df.to_dict("records")

    success = 0
    already_exists = 0
    failed = 0

    start_time = time.time()

    # CPU-friendly number of workers
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = [
            executor.submit(
                download_image,
                row,
                image_dir
            )
            for row in rows
        ]

        for i, future in enumerate(
            as_completed(futures),
            start=1
        ):

            image_id, image_path, status, error = future.result()

            # Find row
            index = df.index[
                df["id"].astype(str) == str(image_id)
            ]

            if len(index) > 0:

                idx = index[0]

                df.at[idx, "image_path"] = image_path
                df.at[idx, "status"] = status
                df.at[idx, "error"] = error

            if status == "success":
                success += 1

            elif status == "already_exists":
                already_exists += 1

            else:
                failed += 1

            if i % 500 == 0 or i == len(rows):
                print(
                    f"Progress: {i}/{len(rows)} | "
                    f"New: {success} | "
                    f"Existing: {already_exists} | "
                    f"Failed: {failed}"
                )

    # Save updated CSV
    df.to_csv(
        csv_path,
        index=False
    )

    elapsed = time.time() - start_time

    print("\nCompleted:", split_name)
    print("Total:", len(df))
    print("New images:", success)
    print("Existing images:", already_exists)
    print("Failed:", failed)
    print(f"Time: {elapsed / 60:.2f} minutes")


def main():

    total_start = time.time()

    for split_name, csv_name in SPLITS.items():
        process_split(
            split_name,
            csv_name
        )

    total_time = time.time() - total_start

    print("\n" + "=" * 50)
    print("20K IMAGE DOWNLOAD COMPLETED")
    print("=" * 50)
    print(f"Total time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()