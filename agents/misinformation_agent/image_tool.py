import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from google.adk.tools import ToolContext


MODEL_NAME = "openai/clip-vit-base-patch32"


processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()


def analyze_image(
    image_path: str,
    tool_context: ToolContext
) -> dict:
    """
    Generate a CLIP image embedding for the supplied image.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not open image: {str(e)}"
        }

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_outputs = model.get_image_features(**inputs)

    # Handle different Transformers output formats
    if hasattr(image_outputs, "pooler_output"):
        embedding = image_outputs.pooler_output

    elif hasattr(image_outputs, "last_hidden_state"):
        embedding = image_outputs.last_hidden_state[:, 0, :]

    else:
        embedding = image_outputs

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    embedding = embedding.cpu().numpy()[0]

    # Store embedding internally in ADK session state
    tool_context.state["image_embedding"] = embedding.tolist()

    return {
        "status": "success",
        "embedding_dimension": int(len(embedding))
    }