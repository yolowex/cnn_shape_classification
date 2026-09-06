import torch
from torch import nn
from collections import defaultdict

import load_dataset
from cnn import Network


# Settings
NUM_CLASSES = 6
CHECKPOINT_PATH = "best_model.pt"

classes = [
    "circle",
    "oval",
    "line",
    "rectangle",
    "square",
    "triangle",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@torch.no_grad()
def test_per_class(model, test_loader):
    model.eval()

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    total_correct = 0
    total_samples = 0

    for inputs, targets in test_loader:
        inputs = inputs.to(DEVICE).float()
        targets = targets.to(DEVICE).long()

        # Supports input shape [batch_size, 32, 32]
        # and [batch_size, 1, 32, 32].
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        logits = model(inputs)
        predictions = logits.argmax(dim=1)

        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

        for target, prediction in zip(targets, predictions):
            target_index = target.item()
            prediction_index = prediction.item()

            total_per_class[target_index] += 1

            if target_index == prediction_index:
                correct_per_class[target_index] += 1

    overall_accuracy = total_correct / total_samples

    print("\nPer-class accuracy")
    print("=" * 45)

    for class_index, class_name in enumerate(classes):
        total = total_per_class[class_index]
        correct = correct_per_class[class_index]

        if total == 0:
            accuracy = 0.0
            print(
                f"{class_name:12s} | "
                f"No samples found"
            )
        else:
            accuracy = correct / total
            print(
                f"{class_name:12s} | "
                f"Correct: {correct:4d} | "
                f"Total: {total:4d} | "
                f"Accuracy: {accuracy:.4f} "
                f"({accuracy * 100:.2f}%)"
            )

    print("=" * 45)
    print(
        f"Overall accuracy: {overall_accuracy:.4f} "
        f"({overall_accuracy * 100:.2f}%)"
    )


def main():
    print(f"Using device: {DEVICE}")

    model = Network().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded model from epoch {checkpoint['epoch']}"
    )

    # This assumes load_dataset.py provides test_loader.
    test_per_class(
        model,
        load_dataset.validation_loader,
    )


if __name__ == "__main__":
    main()
