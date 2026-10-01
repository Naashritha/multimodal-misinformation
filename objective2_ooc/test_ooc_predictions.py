import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from agents.misinformation_agent.ooc_tool import predict_ooc


image_path = r"objective2_ooc\test_images\test.jpg"


tests = [
    "A photograph showing the subject in this image.",
    "A photograph of a completely unrelated event happening in another country."
]


for i, text in enumerate(tests, 1):

    print("\n================================")
    print(f"TEST {i}")
    print("================================")

    print("Text:", text)

    result = predict_ooc(
        text=text,
        image_path=image_path
    )

    print("Result:", result)