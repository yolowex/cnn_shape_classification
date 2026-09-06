import torch
from torch import nn


# Data settings
INPUT_SIZE = (32, 32)
NUM_CLASSES = 6

classes = [
    "circle",
    "oval",
    "line",
    "rectangle",
    "square",
    "triangle",
]


# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [batch_size, 1, 32, 32]
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Output: [batch_size, 32, 16, 16]

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Output: [batch_size, 64, 8, 8]

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Output: [batch_size, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 128 feature maps of size 4x4
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),

            nn.Dropout(p=0.3),

            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        # Dataset images have shape:
        # [batch_size, 32, 32]
        #
        # Conv2d expects:
        # [batch_size, channels, height, width]
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.classifier(x)

        # Return raw logits.
        # Do not apply softmax when using CrossEntropyLoss.
        return x


loss_function = nn.CrossEntropyLoss()


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE).float()
        targets = targets.to(DEVICE).long()

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted_indices = predictions.argmax(dim=1)

        correct_predictions += (
            predicted_indices == targets
        ).sum().item()

        total_predictions += targets.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions

    return average_loss, accuracy


@torch.no_grad()
def evaluate(model, validation_loader):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, targets in validation_loader:
        inputs = inputs.to(DEVICE).float()
        targets = targets.to(DEVICE).long()

        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        total_loss += loss.item()

        predicted_indices = predictions.argmax(dim=1)

        correct_predictions += (
            predicted_indices == targets
        ).sum().item()

        total_predictions += targets.size(0)

    average_loss = total_loss / len(validation_loader)
    accuracy = correct_predictions / total_predictions

    return average_loss, accuracy


def train(model, train_loader, validation_loader):
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_validation_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = "best_model.pt"

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
        )

        scheduler.step(validation_loss)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        # Save whenever this is the best validation accuracy so far.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "classes": classes,
                },
                checkpoint_path,
            )

            save_message = " | Saved best model"
        else:
            save_message = ""

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f} | "
            f"Learning Rate: {current_learning_rate:.6f}"
            f"{save_message}"
        )

    # Restore the best-performing model before returning it.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"\nBest model restored from epoch {checkpoint['epoch']}"
    )
    print(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    return model

if __name__ == "__main__":
    import load_dataset

    model = Network()

    print(f"Using device: {DEVICE}")
    print(model)

    train(
        model,
        load_dataset.train_loader,
        load_dataset.validation_loader,
    )
