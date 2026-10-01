import torch
from transformers import CLIPProcessor, CLIPModel
from google.adk.tools import ToolContext


MODEL_NAME = "openai/clip-vit-base-patch32"


processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()


def get_tensor(output):
    """
    Extract the actual tensor from different
    Transformers output formats.
    """

    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]

    raise TypeError(
        f"Unsupported output type: {type(output)}"
    )


def calculate_consistency(
    text: str,
    image_path: str,
    tool_context: ToolContext
) -> dict:
    """
    Calculate CLIP text-image cosine similarity.
    """

    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not open image: {str(e)}"
        }

    if not text or not text.strip():
        return {
            "status": "error",
            "message": "No text was provided."
        }

    # Process text and image
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():

        text_output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        image_output = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    # Extract actual tensors
    text_features = get_tensor(text_output)
    image_features = get_tensor(image_output)

    # Normalize
    text_features = torch.nn.functional.normalize(
        text_features,
        p=2,
        dim=1
    )

    image_features = torch.nn.functional.normalize(
        image_features,
        p=2,
        dim=1
    )

    # Cosine similarity
    similarity = torch.sum(
        text_features * image_features,
        dim=1
    ).item()

    # Store consistency score internally in ADK state
    tool_context.state["consistency_score"] = float(similarity)

    return {
        "status": "success",
        "consistency_score": float(similarity)
    }