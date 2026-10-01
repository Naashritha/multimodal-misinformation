from misinformation_agent.text_tool import analyze_text
from misinformation_agent.image_tool import analyze_image
from misinformation_agent.consistency_tool import calculate_consistency
from misinformation_agent.fusion_tool import predict_misinformation


# --------------------------------------------------
# Test input
# --------------------------------------------------
TEXT = "samsung lowprofile heir poised to succeed father seen as god"

IMAGE_PATH = r"data\images\test\2eohu1.jpg"


# --------------------------------------------------
# Text analysis
# --------------------------------------------------

print("\nAnalyzing text...")

text_result = analyze_text(TEXT)

print(
    "Text status:",
    text_result["status"]
)

print(
    "Text embedding dimension:",
    text_result["embedding_dimension"]
)


# --------------------------------------------------
# Image analysis
# --------------------------------------------------

print("\nAnalyzing image...")

image_result = analyze_image(IMAGE_PATH)

print(
    "Image status:",
    image_result["status"]
)

print(
    "Image embedding dimension:",
    image_result["embedding_dimension"]
)


# --------------------------------------------------
# Consistency
# --------------------------------------------------

print("\nCalculating text-image consistency...")

consistency_result = calculate_consistency(
    TEXT,
    IMAGE_PATH
)

print(
    "Consistency:",
    consistency_result
)


# --------------------------------------------------
# Fusion prediction
# --------------------------------------------------

print("\nRunning multimodal classifier...")

prediction = predict_misinformation(
    text_result["embedding"],
    image_result["embedding"],
    consistency_result["consistency_score"]
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\n" + "=" * 50)
print("FINAL RESULT")
print("=" * 50)

print(
    "Prediction:",
    prediction["label"]
)

print(
    "Confidence:",
    f"{prediction['confidence']:.4f}"
)

print(
    "Consistency:",
    f"{prediction['consistency_score']:.4f}"
)

print("=" * 50)