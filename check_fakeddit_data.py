import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/fakeddit")

files = [
    "all_train.tsv",
    "all_validate.tsv",
    "all_test_public.tsv"
]

for filename in files:
    path = DATA_DIR / filename

    df = pd.read_csv(path, sep="\t")

    print("\n" + "=" * 70)
    print(filename)

    print("Total samples:", len(df))

    # Check text + image availability
    usable = df[
        df["clean_title"].notna()
        & df["image_url"].notna()
        & (df["hasImage"] == True)
    ]

    print("Text + image samples:", len(usable))

    print("\n2-way label dtype:")
    print(df["2_way_label"].dtype)

    print("\n2-way label values:")
    print(df["2_way_label"].value_counts(dropna=False).sort_index())

    print("\nMissing values:")
    print("clean_title:", df["clean_title"].isna().sum())
    print("image_url:", df["image_url"].isna().sum())
    print("2_way_label:", df["2_way_label"].isna().sum())

    print("\nUsable samples by 2-way label:")
    print(usable["2_way_label"].value_counts(dropna=False).sort_index())