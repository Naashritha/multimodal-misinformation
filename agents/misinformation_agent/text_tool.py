import os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from google.adk.tools import ToolContext


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


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


def analyze_text(
    text: str,
    tool_context: ToolContext
) -> dict:
    """
    Generate a MiniLM embedding for the supplied text.
    """

    if not text or not text.strip():
        return {
            "status": "error",
            "message": "No text was provided."
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(
        outputs,
        inputs["attention_mask"]
    )

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    embedding = embedding.cpu().numpy()[0]

    # Store embedding internally in ADK session state
    tool_context.state["text_embedding"] = embedding.tolist()

    return {
        "status": "success",
        "embedding_dimension": int(len(embedding))
    }