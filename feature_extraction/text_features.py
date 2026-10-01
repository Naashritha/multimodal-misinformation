import pandas as pd
import numpy as np
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Paths
PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("feature_extraction/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModel.from_pretrained(MODEL_NAME)

# CPU
device = torch.device("cpu")
model.to(device)

# Freeze model
model.eval()

for param in model.parameters():
    param.requires_grad = False


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    return torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9
    )


def extract_features(csv_name, output_name):

    print("\n" + "=" * 50)
    print(f"Processing {csv_name}")
    print("=" * 50)

    df = pd.read_csv(PROCESSED_DIR / csv_name)

    texts = df["text"].fillna("").astype(str).tolist()

    embeddings = []

    batch_size = 16

    with torch.no_grad():

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Extracting text features"
        ):

            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            output = model(**encoded)

            batch_embeddings = mean_pooling(
                output,
                encoded["attention_mask"]
            )

            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings,
                p=2,
                dim=1
            )

            embeddings.append(
                batch_embeddings.cpu().numpy()
            )

    embeddings = np.vstack(embeddings)

    output_path = FEATURE_DIR / output_name

    np.save(output_path, embeddings)

    print("\nSaved:", output_path)
    print("Shape:", embeddings.shape)


def main():

    extract_features(
        "train_final.csv",
        "train_text_embeddings.npy"
    )

    extract_features(
        "val_final.csv",
        "val_text_embeddings.npy"
    )

    extract_features(
        "test_final.csv",
        "test_text_embeddings.npy"
    )

    print("\n" + "=" * 50)
    print("TEXT FEATURE EXTRACTION COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()