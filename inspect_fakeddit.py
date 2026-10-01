import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/fakeddit")

for file in DATA_DIR.glob("*.tsv"):
    print("\n" + "=" * 60)
    print("FILE:", file.name)

    df = pd.read_csv(file, sep="\t")

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())