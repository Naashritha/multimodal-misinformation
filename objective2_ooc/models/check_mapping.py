import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = BASE_DIR / "data" / "merged_balanced" / "train.json"
IMAGE_PATH = BASE_DIR / "data" / "embeddings" / "clip_image_embeddings" / "clip_image_embeddings_train.pkl"
TEXT_PATH = BASE_DIR / "data" / "embeddings" / "clip_text_embeddings" / "clip_text_embeddings_train.pkl"


with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]

with open(IMAGE_PATH, "rb") as f:
    image_embeddings = pickle.load(f)

with open(TEXT_PATH, "rb") as f:
    text_embeddings = pickle.load(f)


print("Total annotations:", len(annotations))
print("Image embeddings:", len(image_embeddings))
print("Text embeddings:", len(text_embeddings))

print("\nFirst 10 annotations:\n")

for annotation in annotations[:10]:

    sample_id = annotation["id"]
    image_id = annotation["image_id"]

    print("Annotation:")
    print("  id       :", sample_id)
    print("  image_id :", image_id)
    print("  falsified:", annotation["falsified"])

    print("  image[id] available      :", sample_id in image_embeddings)
    print("  image[image_id] available:", image_id in image_embeddings)
    print("  text[id] available       :", sample_id in text_embeddings)

    print()