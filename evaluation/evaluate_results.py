import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_DIR = Path("models")
EVALUATION_DIR = Path("evaluation")

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    # Load baseline results
    results = pd.read_csv(
        MODEL_DIR / "baseline_comparison.csv"
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(index=False)
    )

    # --------------------------------------------------
    # Save clean results
    # --------------------------------------------------

    results.to_csv(
        EVALUATION_DIR / "final_results.csv",
        index=False
    )

    # --------------------------------------------------
    # Accuracy comparison
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        results["Model"],
        results["Accuracy"]
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Accuracy Comparison")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "accuracy_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # F1 comparison
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        results["Model"],
        results["F1"]
    )

    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.title("F1 Score Comparison")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "f1_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # All metrics comparison
    # --------------------------------------------------

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    x = range(len(results["Model"]))
    width = 0.2

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):

        values = results[metric]

        positions = [
            value + (i - 1.5) * width
            for value in x
        ]

        plt.bar(
            positions,
            values,
            width=width,
            label=metric
        )

    plt.xticks(
        list(x),
        results["Model"],
        rotation=20,
        ha="right"
    )

    plt.ylabel("Score")
    plt.title("Multimodal Model Performance Comparison")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "metrics_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nGenerated files:")

    print(
        "evaluation/final_results.csv"
    )

    print(
        "evaluation/accuracy_comparison.png"
    )

    print(
        "evaluation/f1_comparison.png"
    )

    print(
        "evaluation/metrics_comparison.png"
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()