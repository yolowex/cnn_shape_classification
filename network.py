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
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(32 * 32, (32 * 16) * 2),
            nn.ReLU(),

            nn.Linear((32 * 16) * 2, (32 * 16) // 2),
            nn.ReLU(),

            nn.Linear((32 * 16) // 2, (32 * 16) // 4),
            nn.ReLU(),

            nn.Linear((32 * 16) // 4, (32 * 16) // 4),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear((32 * 16) // 4, NUM_CLASSES)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.layers(x)
        return self.output_layer(x)  # Raw logits


loss_function = nn.CrossEntropyLoss()


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(DEVICE)
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
        inputs = inputs.to(DEVICE)
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

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    return model


if __name__ == "__main__":
    import load_dataset

    model = Network()

    train(
        model,
        load_dataset.train_loader,
        load_dataset.validation_loader,
    )
