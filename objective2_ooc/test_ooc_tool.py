from agents.misinformation_agent.ooc_tool import predict_ooc


# Replace these with an actual NewsCLIPpings image path
# and its corresponding caption.

text = "A major event happened in the city."

image_path = r"PUT_YOUR_IMAGE_PATH_HERE"


result = predict_ooc(
    text=text,
    image_path=image_path
)

print("\n================================")
print("OOC Prediction Test")
print("================================")

print("Text:", text)
print("Image:", image_path)

print("\nResult:")
print(result)