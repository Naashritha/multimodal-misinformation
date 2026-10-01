import csv
import os
import joblib
import numpy as np

from misinformation_agent.text_tool import analyze_text
from misinformation_agent.image_tool import analyze_image
from misinformation_agent.consistency_tool import calculate_consistency
from misinformation_agent.fusion_tool import predict_misinformation


TEST_FILE = "data/processed/test_final.csv"


def evaluate():

    y_true = []
    y_pred = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for i, row in enumerate(reader):

            print(f"\nTesting sample {i + 1}/{sum(1 for _ in open(TEST_FILE, encoding='utf-8')) - 1}")

            text = row["text"]
            image_path = row["image_path"]
            true_label = int(row["label"])

            print("ID:", row["id"])
            print("Text:", text[:100])
            print("Image:", image_path)

            # ------------------------------------------------
            # 1. Text embedding
            # ------------------------------------------------

            text_result = analyze_text(text)

            if text_result["status"] != "success":
                print("Text analysis failed.")
                continue

            # ------------------------------------------------
            # 2. Image embedding
            # ------------------------------------------------

            image_result = analyze_image(image_path)

            if image_result["status"] != "success":
                print("Image analysis failed.")
                continue

            # ------------------------------------------------
            # 3. Consistency
            # ------------------------------------------------

            consistency_result = calculate_consistency(
                text,
                image_path
            )

            if consistency_result["status"] != "success":
                print("Consistency calculation failed.")
                continue

            # ------------------------------------------------
            # 4. Fusion prediction
            # ------------------------------------------------

            prediction_result = predict_misinformation(
                text_result["embedding"],
                image_result["embedding"],
                consistency_result["consistency_score"]
            )

            if prediction_result["status"] != "success":
                print("Prediction failed.")
                continue

            prediction = prediction_result["prediction"]

            y_true.append(true_label)
            y_pred.append(prediction)

            print("True label:", true_label)
            print("Predicted:", prediction)

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    if len(y_true) == 0:
        print("\nNo samples were successfully evaluated.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("\n" + "=" * 50)
    print("ADK MULTIMODAL EVALUATION RESULTS")
    print("=" * 50)

    print(f"Samples evaluated : {len(y_true)}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print(f"Precision         : {precision * 100:.2f}%")
    print(f"Recall            : {recall * 100:.2f}%")
    print(f"F1 Score          : {f1 * 100:.2f}%")

    print("\nConfusion Matrix")
    print("----------------")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")


if __name__ == "__main__":
    evaluate()